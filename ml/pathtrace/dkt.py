"""Deep Knowledge Tracing evaluator and the learning-gain metric.

The DKT model (Piech et al., 2015) is an LSTM over the attempt history that
predicts, for every concept, the probability of answering it correctly next.
It is trained on training students only and is never used to train a
recommender, so a policy cannot win by exploiting it.

Learning gain of a path for one example:

1. E_before: the mean predicted chance of answering each of the example's N
   candidates correctly, given the history.
2. The simulated student works through the path, ``attempts`` attempts per
   concept, with each answer sampled from DKT and fed back into it.
3. E_after: the same mean over the same N candidates, averaged over
   ``rollouts`` simulations.

    E_p = (E_after - E_before) / (1 - E_before)

All methods are scored with the same random numbers (same seed and batch
order), which makes differences between methods less noisy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pathtrace import metrics
from pathtrace.batching import make_batch
from pathtrace.preprocess import Examples, Interactions, Processed
from pathtrace.utils import get_device, save_json, set_seed


@dataclass
class DKTConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.2
    chunk_len: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 10
    patience: int = 2
    seed: int = 0
    device: str = "auto"


class DKT(nn.Module):
    def __init__(self, num_concepts: int, cfg: DKTConfig):
        super().__init__()
        self.num_concepts = num_concepts
        # Interaction id = 2 * concept + correct; id 0/1 (concept 0) is padding.
        self.embed = nn.Embedding(2 * (num_concepts + 1), cfg.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.hidden_dim, num_concepts + 1)

    def _inputs(self, concepts: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        return self.embed(2 * concepts + correct.long())

    def sequence_logits(self, concepts: torch.Tensor, correct: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Logits for every concept after each step, shape (B, T, C + 1)."""
        packed = pack_padded_sequence(
            self._inputs(concepts, correct), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=concepts.size(1))
        return self.out(self.dropout(out))

    def encode(self, concepts: torch.Tensor, correct: torch.Tensor, lengths: torch.Tensor):
        """LSTM state after each history, ((1, B, H), (1, B, H))."""
        packed = pack_padded_sequence(
            self._inputs(concepts, correct), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.lstm(packed)
        return state

    def step(self, concepts: torch.Tensor, correct: torch.Tensor, state):
        """Feed one attempt per row; returns the new state."""
        _, state = self.lstm(self._inputs(concepts, correct).unsqueeze(1), state)
        return state

    def probs(self, state) -> torch.Tensor:
        return torch.sigmoid(self.out(state[0][-1]))


# --------------------------------------------------------------------------- training


def _chunks(inter: Interactions, students: np.ndarray, chunk_len: int) -> tuple[np.ndarray, np.ndarray]:
    starts, lengths = [], []
    for s in students:
        lo, hi = int(inter.offsets[s]), int(inter.offsets[s + 1])
        for start in range(lo, hi, chunk_len):
            length = min(chunk_len, hi - start)
            if length >= 2:
                starts.append(start)
                lengths.append(length)
    return np.asarray(starts, np.int64), np.asarray(lengths, np.int64)


def _chunk_batch(inter: Interactions, starts: np.ndarray, lengths: np.ndarray, device: torch.device):
    width = int(lengths.max())
    concepts = np.zeros((len(starts), width), np.int64)
    correct = np.zeros((len(starts), width), np.int64)
    for row, (start, length) in enumerate(zip(starts, lengths)):
        concepts[row, :length] = inter.concept[start : start + length]
        correct[row, :length] = inter.correct[start : start + length]
    return (
        torch.from_numpy(concepts).to(device),
        torch.from_numpy(correct).to(device),
        torch.from_numpy(lengths),
    )


def _next_step_outputs(model: DKT, concepts, correct, lengths) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.sequence_logits(concepts, correct, lengths)[:, :-1]
    picked = logits.gather(2, concepts[:, 1:].unsqueeze(-1)).squeeze(-1)
    steps = torch.arange(concepts.size(1) - 1, device=concepts.device)
    mask = steps.unsqueeze(0) < (lengths.to(concepts.device) - 1).unsqueeze(1)
    return picked[mask], correct[:, 1:][mask].float()


@torch.no_grad()
def evaluate_dkt(model: DKT, inter: Interactions, students: np.ndarray, cfg: DKTConfig, device) -> dict:
    model.eval()
    starts, lengths = _chunks(inter, students, cfg.chunk_len)
    scores, labels = [], []
    for i in range(0, len(starts), cfg.batch_size):
        logit, label = _next_step_outputs(model, *_chunk_batch(inter, starts[i : i + cfg.batch_size], lengths[i : i + cfg.batch_size], device))
        scores.append(logit.cpu().numpy())
        labels.append(label.cpu().numpy())
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(scores), torch.from_numpy(labels)).item()
    return {"auc": metrics.auc(labels, scores), "loss": loss, "predictions": int(len(labels))}


def train_dkt(proc: Processed, cfg: DKTConfig, out_dir: str | Path, log: Callable[[str], None] = print) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = DKT(proc.inter.num_concepts, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    starts, lengths = _chunks(proc.inter, proc.splits["train"], cfg.chunk_len)
    rng = np.random.default_rng(cfg.seed)
    log(f"DKT: {len(starts):,} training chunks, device {device}")

    best_auc, best_epoch, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(starts))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            logit, label = _next_step_outputs(model, *_chunk_batch(proc.inter, starts[idx], lengths[idx], device))
            loss = F.binary_cross_entropy_with_logits(logit, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val = evaluate_dkt(model, proc.inter, proc.splits["val"], cfg, device)
        history.append({"epoch": epoch, "val_auc": val["auc"], "val_loss": val["loss"]})
        log(f"  DKT epoch {epoch:2d}  val AUC {val['auc']:.4f}  loss {val['loss']:.4f}")
        if val["auc"] > best_auc + 1e-4:
            best_auc, best_epoch = val["auc"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {"model_state": model.state_dict(), "config": dataclasses.asdict(cfg), "num_concepts": proc.inter.num_concepts},
        out / "dkt.pt",
    )
    summary = {
        "best_epoch": best_epoch,
        "history": history,
        "val": evaluate_dkt(model, proc.inter, proc.splits["val"], cfg, device),
        "test": evaluate_dkt(model, proc.inter, proc.splits["test"], cfg, device),
    }
    save_json(summary, out / "dkt_summary.json")
    return summary


def load_dkt(path: str | Path, device: torch.device) -> DKT:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    known = {f.name for f in dataclasses.fields(DKTConfig)}
    cfg = DKTConfig(**{k: v for k, v in ckpt["config"].items() if k in known})
    model = DKT(ckpt["num_concepts"], cfg)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- learning gain


@torch.no_grad()
def learning_gain(
    model: DKT,
    inter: Interactions,
    ex: Examples,
    paths: dict[str, np.ndarray],
    *,
    attempts: int = 3,
    rollouts: int = 8,
    max_history: int = 200,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-example E_before, E_after and E_p for each method's paths (candidate slots).

    The history is encoded once per batch and shared by all methods, and every
    method sees the same uniform draws for its simulated answers.
    """
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    before = np.empty(len(ex))
    after = {name: np.empty(len(ex)) for name in paths}
    for start in range(0, len(ex), batch_size):
        idx = np.arange(start, min(start + batch_size, len(ex)))
        batch = make_batch(inter, ex, idx, max_history).to(device)
        encoded = model.encode(batch.hist_concepts, batch.hist_correct, batch.lengths)
        cands = batch.candidates
        before[idx] = model.probs(encoded).gather(1, cands).mean(dim=1).cpu().numpy()
        cands_rep = cands.repeat_interleave(rollouts, dim=0)
        for name, method_paths in paths.items():
            generator.manual_seed(seed + start)
            state = tuple(s.repeat_interleave(rollouts, dim=1) for s in encoded)
            slots = torch.from_numpy(method_paths[idx]).to(device)
            path_concepts = cands.gather(1, slots).repeat_interleave(rollouts, dim=0)
            for t in range(path_concepts.size(1)):
                concept = path_concepts[:, t]
                for _ in range(attempts):
                    p = model.probs(state).gather(1, concept.unsqueeze(1)).squeeze(1)
                    uniform = torch.rand(p.shape, generator=generator, device=device)
                    state = model.step(concept, (uniform < p).float(), state)
            mean_after = model.probs(state).gather(1, cands_rep).mean(dim=1)
            after[name][idx] = mean_after.view(len(idx), rollouts).mean(dim=1).cpu().numpy()
    headroom = np.maximum(1.0 - before, 1e-6)
    return {
        name: {"before": before, "after": after[name], "gain": (after[name] - before) / headroom} for name in paths
    }


# --------------------------------------------------------------------------- command


def collect_paths(proc: Processed, split: str, runs: list[Path], bkt_params: Path | None = None) -> dict[str, np.ndarray]:
    """Paths to score: random, the students' own next concepts, saved predictions, and the BKT oracle."""
    from pathtrace import bkt

    ex = proc.examples[split]
    k = proc.data_cfg.path_len
    rng = np.random.default_rng(0)
    paths = {
        "random": np.argsort(rng.random(ex.candidates.shape), axis=1)[:, :k],
        "students": ex.target_slots,
    }
    for run in runs:
        for f in sorted(Path(run).glob(f"preds*_{split}.npy")):
            name = f.stem[len("preds_") : -len(f"_{split}")] if f.stem != f"preds_{split}" else Path(run).name
            paths[name] = np.load(f)
    if bkt_params is not None:
        params = bkt.BKTParams.load(bkt_params)
        gains = bkt.expected_gain(bkt.candidate_mastery(proc.inter, ex, params), ex.candidates, params)
        paths["bkt_oracle"] = bkt.oracle_paths(gains, k)
    return paths


def gain_report(
    proc: Processed,
    model: DKT,
    split: str,
    paths: dict[str, np.ndarray],
    *,
    max_examples: int | None = None,
    n_boot: int = 1000,
    **kwargs,
) -> dict:
    ex = proc.examples[split]
    rows = np.arange(len(ex))
    if max_examples is not None and max_examples < len(ex):
        rows = np.sort(np.random.default_rng(0).choice(len(ex), max_examples, replace=False))
    subset = Examples(**{f.name: getattr(ex, f.name)[rows] for f in dataclasses.fields(Examples)})
    results = learning_gain(model, proc.inter, subset, {name: p[rows] for name, p in paths.items()}, **kwargs)
    return {
        "examples": int(len(rows)),
        "methods": {
            name: {
                "before": float(r["before"].mean()),
                "after": float(r["after"].mean()),
                "gain": metrics.summarize({"gain": r["gain"]}, n_boot)["gain"],
            }
            for name, r in results.items()
        },
    }
