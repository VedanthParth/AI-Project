"""Policy-gradient fine-tuning against the BKT reward (paper Section 5.2, corrected).

Reward of a path P for one example (``RewardTables``):

    R(P) = sum_{c in P} (1 - m_c) * p_learn_c  -  lambda_v * V(P)

with per-concept BKT parameters (``bkt``) and V(P) the number of path
concepts started before all their inferred prerequisites (``prereq``).

Loss for a batch (``finetune``):

    L = -mean_b (R(tau_b) - baseline_b) * sum_t log pi(a_bt | s_bt)
        - beta_H * mean per-step entropy
        + lambda_sup * teacher-forced path cross-entropy

Baselines:

- "greedy" (default, self-critical): the reward of the greedy path for the
  same example. It depends on the parameters but not on the sampled actions,
  so the gradient estimate is unbiased (Proposition 1).
- "ema": a moving average of batch rewards, applied before it is updated
  with the current batch, and warm-started from the greedy reward of the
  first batch rather than 0. The paper's Algorithm 1 updated it first, which
  makes the baseline depend on the sampled actions.

``oracle_paths`` gives the exact best path under R (all K-permutations of the
N candidates), the upper bound every method is compared against.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from pathtrace import bkt, metrics, prereq
from pathtrace.batching import iterate_batches
from pathtrace.preprocess import Processed
from pathtrace.train import load_checkpoint
from pathtrace.utils import get_device, save_json, set_seed


@dataclass
class RLConfig:
    lr: float = 5e-4
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 3
    entropy_weight: float = 0.01
    sup_weight: float = 0.5
    violation_weight: float = 0.2
    baseline: str = "greedy"
    ema_momentum: float = 0.9
    grad_clip: float = 5.0
    max_history: int = 100
    bootstrap: int = 1000
    seed: int = 0
    device: str = "auto"


# --------------------------------------------------------------------------- reward


@dataclass
class RewardTables:
    """Everything the reward needs for one split, indexed by example."""

    gains: np.ndarray  # (E, N) expected BKT gain per candidate
    always: np.ndarray  # (E, N) candidate always violates a prerequisite
    need: np.ndarray  # (E, N, N) candidate i needs candidate j earlier in the path
    violation_weight: float

    def violations(self, idx: np.ndarray, paths: np.ndarray) -> np.ndarray:
        return prereq.path_violations(paths, self.always[idx], self.need[idx])

    def reward(self, idx: np.ndarray, paths: np.ndarray) -> np.ndarray:
        gain = bkt.path_reward(self.gains[idx], paths)
        return gain - self.violation_weight * self.violations(idx, paths)


def build_reward_tables(
    proc: Processed, split: str, params: bkt.BKTParams, graph: prereq.PrereqGraph, violation_weight: float
) -> RewardTables:
    ex = proc.examples[split]
    gains = bkt.expected_gain(bkt.candidate_mastery(proc.inter, ex, params), ex.candidates, params)
    always, need = prereq.example_constraints(proc.inter, ex, graph)
    return RewardTables(gains.astype(np.float64), always, need, violation_weight)


def oracle_paths(tables: RewardTables, k: int, chunk: int = 256) -> np.ndarray:
    """Exact reward-maximising K-path for every example (brute force over K-permutations)."""
    n = tables.gains.shape[1]
    perms = np.array(list(permutations(range(n), k)))  # (P, K)
    need_count = tables.need.sum(axis=2)
    best = np.empty((len(tables.gains), k), dtype=np.int64)
    for start in range(0, len(tables.gains), chunk):
        sl = slice(start, start + chunk)
        gains, always, need, count = tables.gains[sl], tables.always[sl], tables.need[sl], need_count[sl]
        total = gains[:, perms].sum(axis=2)  # (B, P)
        violations = np.zeros_like(total)
        for t in range(k):
            slot = perms[:, t]
            satisfied = need[:, slot[:, None], perms[:, :t]].sum(axis=2) if t else 0
            violations += always[:, slot] | ((count[:, slot] - satisfied) > 0)
        score = total - tables.violation_weight * violations
        best[sl] = perms[score.argmax(axis=1)]
    return best


# --------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_policy(model, proc: Processed, split: str, tables: RewardTables, oracle_reward: np.ndarray, cfg: RLConfig, device, n_boot: int = 0) -> tuple[dict, np.ndarray]:
    model.eval()
    ex = proc.examples[split]
    k = proc.data_cfg.path_len
    preds = []
    for batch in iterate_batches(proc.inter, ex, cfg.batch_size, cfg.max_history):
        preds.append(model.recommend(batch.to(device), k).actions.cpu().numpy())
    pred = np.concatenate(preds)
    idx = np.arange(len(ex))
    reward = tables.reward(idx, pred)
    result = {
        "n": len(ex),
        "metrics": metrics.summarize(metrics.per_example(pred, ex.target_slots, ex.first_group), n_boot, cfg.seed),
        "reward": metrics.summarize({"reward": reward}, n_boot, cfg.seed)["reward"],
        "oracle_reward": float(oracle_reward.mean()),
        "share_of_oracle": float(reward.mean() / oracle_reward.mean()) if oracle_reward.mean() else float("nan"),
        "violations_per_path": float(tables.violations(idx, pred).mean()),
    }
    return result, pred


# --------------------------------------------------------------------------- training


def finetune(
    proc: Processed,
    checkpoint: str | Path,
    params: bkt.BKTParams,
    graph: prereq.PrereqGraph,
    cfg: RLConfig,
    out_dir: str | Path,
    log: Callable[[str], None] = print,
) -> dict:
    if cfg.baseline not in {"greedy", "ema"}:
        raise ValueError(f"baseline must be 'greedy' or 'ema', got {cfg.baseline!r}")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = proc.data_cfg.path_len
    model, ckpt = load_checkpoint(checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    log("building reward tables (BKT gains and prerequisite constraints)")
    tables = {split: build_reward_tables(proc, split, params, graph, cfg.violation_weight) for split in ("train", "val", "test")}
    oracle = {}
    for split in ("val", "test"):
        idx = np.arange(len(proc.examples[split]))
        oracle[split] = tables[split].reward(idx, oracle_paths(tables[split], k))

    start_val, _ = evaluate_policy(model, proc, "val", tables["val"], oracle["val"], cfg, device)
    log(f"supervised start: val reward {start_val['reward']['mean']:.4f} ({start_val['share_of_oracle']:.1%} of oracle)")
    best_reward, best_epoch = start_val["reward"]["mean"], 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = [{"epoch": 0, "val_reward": best_reward, "val_ndcg": start_val["metrics"]["ndcg"]["mean"]}]
    ema = None
    rng = np.random.default_rng(cfg.seed)
    train_ex = proc.examples["train"]

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        started = time.time()
        sums = {"sampled_reward": 0.0, "greedy_reward": 0.0, "entropy": 0.0}
        seen = 0
        for batch in iterate_batches(proc.inter, train_ex, cfg.batch_size, cfg.max_history, rng=rng):
            b = batch.to(device)
            idx = batch.index
            state, cand = model.encode_student(b), model.encode_candidates(b.candidates)
            sample = model.decode(state, cand, k, "sample")
            sampled_reward = tables["train"].reward(idx, sample.actions.cpu().numpy())
            with torch.no_grad():
                greedy = model.decode(state, cand, k, "greedy")
            greedy_reward = tables["train"].reward(idx, greedy.actions.cpu().numpy())
            if cfg.baseline == "greedy":
                baseline = greedy_reward
            else:
                if ema is None:
                    ema = greedy_reward.mean()  # warm start that doesn't depend on the samples
                baseline = np.full(len(idx), ema)
                ema = cfg.ema_momentum * ema + (1 - cfg.ema_momentum) * sampled_reward.mean()
            advantage = torch.as_tensor(sampled_reward - baseline, dtype=torch.float32, device=device)
            pg_loss = -(advantage * sample.log_probs.sum(dim=1)).mean()
            entropy = -(sample.probs * torch.log(sample.probs.clamp_min(1e-12))).sum(dim=-1).mean()
            loss = pg_loss - cfg.entropy_weight * entropy
            if cfg.sup_weight > 0:
                teacher = model.decode(state, cand, k, "teacher", target_slots=b.target_slots)
                loss = loss + cfg.sup_weight * model.step_loss(teacher.logits, b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            seen += len(idx)
            sums["sampled_reward"] += sampled_reward.sum()
            sums["greedy_reward"] += greedy_reward.sum()
            sums["entropy"] += entropy.item() * len(idx)

        val, _ = evaluate_policy(model, proc, "val", tables["val"], oracle["val"], cfg, device)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            **{f"train_{name}": total / seen for name, total in sums.items()},
            "val_reward": val["reward"]["mean"],
            "val_share_of_oracle": val["share_of_oracle"],
            "val_violations": val["violations_per_path"],
            "val_ndcg": val["metrics"]["ndcg"]["mean"],
        }
        history.append(record)
        save_json(history, out / "history.json")
        log(
            f"RL epoch {epoch:2d}  train reward sampled {record['train_sampled_reward']:.4f} greedy {record['train_greedy_reward']:.4f}"
            f"  entropy {record['train_entropy']:.3f}  |  val reward {val['reward']['mean']:.4f}"
            f" ({val['share_of_oracle']:.1%} of oracle)  ndcg {record['val_ndcg']:.4f}  ({record['seconds']}s)"
        )
        if val["reward"]["mean"] > best_reward + 1e-5:
            best_reward, best_epoch = val["reward"]["mean"], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            log(f"early stop: no validation reward gain for {cfg.patience} epochs")
            break

    model.load_state_dict(best_state)
    torch.save({**ckpt, "model_state": model.state_dict(), "rl_config": dataclasses.asdict(cfg)}, out / "model.pt")
    results = {
        "best_epoch": best_epoch,
        "start": start_val,
        "chance": {
            split: metrics.split_chance(proc.data_cfg.num_candidates, k, proc.examples[split].first_group)
            for split in ("val", "test")
        },
        "splits": {},
    }
    for split in ("val", "test"):
        result, pred = evaluate_policy(model, proc, split, tables[split], oracle[split], cfg, device, n_boot=cfg.bootstrap)
        results["splits"][split] = result
        np.save(out / f"preds_{split}.npy", pred)
    save_json(results, out / "metrics.json")
    return results

