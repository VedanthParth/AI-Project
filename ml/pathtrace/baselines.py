"""Baselines that rank the same candidate sets as the pointer network.

Each baseline scores every candidate; the top K by score (ties broken at
random) form its path. Everything a baseline learns comes from the training
split only.

- random:        uniform scores (should match ``metrics.chance``)
- popularity:    how many training students attempted the concept
- markov:        transitions between consecutive new concepts in training
                 students' first-attempt order, from the last few concepts of
                 the history (more recent ones weigh more)
- gru_next_item: GRU4Rec-style GRU over the history with a softmax over the
                 whole vocabulary, trained to predict the next K new concepts
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from pathtrace import metrics
from pathtrace.batching import Batch, iterate_batches
from pathtrace.config import ModelConfig, TrainConfig
from pathtrace.model import NEG_INF
from pathtrace.preprocess import Processed, first_attempts
from pathtrace.utils import get_device, save_json, set_seed


class Baseline:
    name = "baseline"

    def fit(self, proc: Processed) -> None:
        """Learn from ``proc.splits['train']`` / ``proc.examples['train']`` only."""

    def score(self, proc: Processed, split: str) -> np.ndarray:
        """Scores for each candidate of each example in ``split``, shape (E, N)."""
        raise NotImplementedError


class RandomBaseline(Baseline):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, proc: Processed, split: str) -> np.ndarray:
        return np.random.default_rng(self.seed).random(proc.examples[split].candidates.shape)


class PopularityBaseline(Baseline):
    name = "popularity"

    def score(self, proc: Processed, split: str) -> np.ndarray:
        return proc.popularity[proc.examples[split].candidates]


class MarkovBaseline(Baseline):
    name = "markov"

    def __init__(self, order: int = 3, decay: float = 0.5):
        self.order = order
        self.decay = decay

    def fit(self, proc: Processed) -> None:
        size = proc.inter.num_concepts + 1
        counts = np.zeros((size, size))
        for s in proc.splits["train"]:
            distinct, _ = first_attempts(proc.inter.sequence(s)[0])
            np.add.at(counts, (distinct[:-1], distinct[1:]), 1.0)
        self.transition = counts / (counts.sum(axis=1, keepdims=True) + 1.0)
        # Tiny popularity term so that unseen transitions still get an order.
        self.backoff = 1e-6 * proc.popularity / max(proc.popularity.max(), 1.0)

    def score(self, proc: Processed, split: str) -> np.ndarray:
        ex = proc.examples[split]
        out = np.empty(ex.candidates.shape)
        last_student, distinct, firsts = -1, None, None
        for i in range(len(ex)):
            if ex.student[i] != last_student:
                last_student = ex.student[i]
                distinct, firsts = first_attempts(proc.inter.sequence(last_student)[0])
            seen = np.searchsorted(firsts, ex.cut[i])  # distinct concepts in the history
            recent = distinct[max(0, seen - self.order) : seen][::-1]
            weights = self.decay ** np.arange(len(recent))
            cands = ex.candidates[i]
            out[i] = weights @ self.transition[np.ix_(recent, cands)] + self.backoff[cands]
        return out


class _NextItemNet(nn.Module):
    def __init__(self, num_concepts: int, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(num_concepts + 1, cfg.embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout)
        self.gru = nn.GRU(cfg.embed_dim + 1, cfg.hidden_dim, batch_first=True)
        self.out = nn.Linear(cfg.hidden_dim, num_concepts + 1)

    def forward(self, batch: Batch) -> torch.Tensor:
        x = torch.cat([self.dropout(self.embed(batch.hist_concepts)), batch.hist_correct.unsqueeze(-1)], dim=-1)
        packed = pack_padded_sequence(x, batch.lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        logits = self.out(self.dropout(h_n[-1]))
        pad = torch.zeros_like(logits, dtype=torch.bool)
        pad[:, 0] = True
        return logits.masked_fill(pad, NEG_INF)


class GRUNextItemBaseline(Baseline):
    name = "gru_next_item"

    def __init__(self, model_cfg: ModelConfig, cfg: TrainConfig, log: Callable[[str], None] = print):
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.log = log
        self.device = get_device(cfg.device)

    def fit(self, proc: Processed) -> None:
        set_seed(self.cfg.seed)
        self.net = _NextItemNet(proc.inter.num_concepts, self.model_cfg).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        rng = np.random.default_rng(self.cfg.seed)
        k = proc.data_cfg.path_len
        best, best_epoch, best_state = -np.inf, 0, None
        for epoch in range(1, self.cfg.max_epochs + 1):
            self.net.train()
            for batch in iterate_batches(
                proc.inter, proc.examples["train"], self.cfg.batch_size, self.cfg.max_history, rng=rng
            ):
                b = batch.to(self.device)
                log_probs = torch.log_softmax(self.net(b), dim=-1)
                loss = -log_probs.gather(1, b.targets).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                optimizer.step()
            val = evaluate_baseline(self, proc, "val", k, seed=self.cfg.seed)[0]
            score = val["metrics"][self.cfg.select_metric]["mean"]
            self.log(f"  gru_next_item epoch {epoch:3d}  val {self.cfg.select_metric} {score:.4f}")
            if score > best + 1e-4:
                best, best_epoch = score, epoch
                best_state = {name: t.detach().clone() for name, t in self.net.state_dict().items()}
            elif epoch - best_epoch >= self.cfg.patience:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    @torch.no_grad()
    def score(self, proc: Processed, split: str) -> np.ndarray:
        self.net.eval()
        ex = proc.examples[split]
        chunks = []
        for batch in iterate_batches(proc.inter, ex, self.cfg.batch_size, self.cfg.max_history):
            b = batch.to(self.device)
            chunks.append(self.net(b).gather(1, b.candidates).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros(ex.candidates.shape)


def evaluate_baseline(
    baseline: Baseline, proc: Processed, split: str, k: int, n_boot: int = 0, seed: int = 0
) -> tuple[dict, np.ndarray]:
    ex = proc.examples[split]
    pred = metrics.top_k(baseline.score(proc, split), k, np.random.default_rng(seed))
    values = metrics.per_example(pred, ex.target_slots, ex.first_group)
    return {"n": len(ex), "metrics": metrics.summarize(values, n_boot, seed)}, pred


BASELINES = ("random", "popularity", "markov", "gru_next_item")


def make_baseline(name: str, model_cfg: ModelConfig, cfg: TrainConfig, log: Callable[[str], None]) -> Baseline:
    if name == "random":
        return RandomBaseline(seed=cfg.seed)
    if name == "popularity":
        return PopularityBaseline()
    if name == "markov":
        return MarkovBaseline()
    if name == "gru_next_item":
        return GRUNextItemBaseline(model_cfg, cfg, log)
    raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")


def run_baselines(
    proc: Processed,
    names: list[str],
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    out_dir: str | Path,
    log: Callable[[str], None] = print,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = proc.data_cfg.path_len
    results: dict = {"chance": metrics.chance(proc.data_cfg.num_candidates, k), "baselines": {}}
    for name in names:
        log(f"baseline {name}")
        baseline = make_baseline(name, model_cfg, cfg, log)
        baseline.fit(proc)
        results["baselines"][name] = {}
        for split in ("val", "test"):
            result, pred = evaluate_baseline(baseline, proc, split, k, n_boot=cfg.bootstrap, seed=cfg.seed)
            results["baselines"][name][split] = result
            np.save(out / f"preds_{name}_{split}.npy", pred)
        save_json(results, out / "baselines.json")
    return results
