"""Supervised training (path cross-entropy + lambda_KT * KT loss) and evaluation."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pathtrace import metrics
from pathtrace.batching import iterate_batches
from pathtrace.config import ModelConfig, TrainConfig, dataclass_from_dict
from pathtrace.model import PathRecommender
from pathtrace.preprocess import Processed
from pathtrace.utils import get_device, save_json, set_seed


@torch.no_grad()
def evaluate_model(
    model: PathRecommender,
    proc: Processed,
    split: str,
    cfg: TrainConfig,
    device: torch.device,
    n_boot: int = 0,
) -> tuple[dict, np.ndarray]:
    """Greedy-decoded ranking metrics, teacher-forced path loss and KT-head quality."""
    model.eval()
    ex = proc.examples[split]
    k = proc.data_cfg.path_len
    if len(ex) == 0:
        return {"n": 0}, np.zeros((0, k), dtype=np.int64)
    preds, kt_scores, kt_labels = [], [], []
    path_loss_sum = 0.0
    for batch in iterate_batches(proc.inter, ex, cfg.batch_size, cfg.max_history):
        b = batch.to(device)
        state = model.encode_student(b)
        cand = model.encode_candidates(b.candidates)
        preds.append(model.decode(state, cand, k, "greedy").actions.cpu().numpy())
        teacher = model.decode(state, cand, k, "teacher", target_slots=b.target_slots)
        path_loss_sum += F.cross_entropy(
            teacher.logits.reshape(-1, teacher.logits.size(-1)), b.target_slots.reshape(-1), reduction="sum"
        ).item()
        logits = model.kt_logits(b)
        mask = model.kt_mask(b, logits.device)
        kt_scores.append(logits[mask].cpu().numpy())
        kt_labels.append(b.hist_correct[:, 1:][mask].cpu().numpy())

    pred = np.concatenate(preds)
    scores, labels = np.concatenate(kt_scores), np.concatenate(kt_labels)
    kt_loss = F.binary_cross_entropy_with_logits(torch.from_numpy(scores), torch.from_numpy(labels)).item()
    result = {
        "n": len(ex),
        "metrics": metrics.summarize(metrics.per_example(pred, ex.target_slots), n_boot, cfg.seed),
        "path_loss": path_loss_sum / (len(ex) * k),
        "kt_loss": kt_loss,
        "kt_auc": metrics.auc(labels, scores),
    }
    return result, pred


def save_checkpoint(path: Path, model: PathRecommender, model_cfg: ModelConfig, cfg: TrainConfig, proc: Processed) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": dataclasses.asdict(model_cfg),
            "train_config": dataclasses.asdict(cfg),
            "data_config": dataclasses.asdict(proc.data_cfg),
            "num_concepts": proc.inter.num_concepts,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[PathRecommender, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PathRecommender(ckpt["num_concepts"], dataclass_from_dict(ModelConfig, ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def train_model(
    proc: Processed,
    model_cfg: ModelConfig,
    cfg: TrainConfig,
    out_dir: str | Path,
    log: Callable[[str], None] = print,
) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = proc.data_cfg.path_len
    train = proc.examples["train"]
    if len(train) == 0:
        raise ValueError("no training examples; check the data and DataConfig")

    model = PathRecommender(proc.inter.num_concepts, model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"{n_params:,} parameters, {len(train):,} training examples, device {device}")

    best_score, best_epoch, best_state = -np.inf, 0, None
    history: list[dict] = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        started = time.time()
        sums = {"loss": 0.0, "path_loss": 0.0, "kt_loss": 0.0}
        seen = 0
        for batch in iterate_batches(proc.inter, train, cfg.batch_size, cfg.max_history, rng=rng):
            b = batch.to(device)
            path_loss = model.path_loss(b, k)
            kt_loss = model.kt_loss(b) if model_cfg.kt_weight > 0 else path_loss.new_zeros(())
            loss = path_loss + model_cfg.kt_weight * kt_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            size = len(b.index)
            seen += size
            sums["loss"] += loss.item() * size
            sums["path_loss"] += path_loss.item() * size
            sums["kt_loss"] += kt_loss.item() * size

        val, _ = evaluate_model(model, proc, "val", cfg, device)
        score = val["metrics"][cfg.select_metric]["mean"]
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            **{f"train_{name}": total / seen for name, total in sums.items()},
            "val_path_loss": val["path_loss"],
            "val_kt_loss": val["kt_loss"],
            "val_kt_auc": val["kt_auc"],
            **{f"val_{name}": val["metrics"][name]["mean"] for name in metrics.METRICS},
        }
        history.append(record)
        save_json(history, out / "history.json")
        log(
            f"epoch {epoch:3d}  train path {record['train_path_loss']:.4f} kt {record['train_kt_loss']:.4f}"
            f"  |  val path {val['path_loss']:.4f}  {cfg.select_metric} {score:.4f}"
            f"  hr {record['val_hit_rate']:.4f}  kt-auc {val['kt_auc']:.4f}  ({record['seconds']}s)"
        )
        if score > best_score + 1e-4:
            best_score, best_epoch = score, epoch
            best_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
            save_checkpoint(out / "model.pt", model, model_cfg, cfg, proc)
        elif epoch - best_epoch >= cfg.patience:
            log(f"early stop: no validation {cfg.select_metric} gain for {cfg.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    results = {
        "best_epoch": best_epoch,
        "parameters": n_params,
        "chance": metrics.chance(proc.data_cfg.num_candidates, k),
        "splits": {},
    }
    for split in ("val", "test"):
        result, pred = evaluate_model(model, proc, split, cfg, device, n_boot=cfg.bootstrap)
        results["splits"][split] = result
        np.save(out / f"preds_{split}.npy", pred)
    save_json(results, out / "metrics.json")
    return results
