"""Ranking metrics for recommended paths, chance levels, and bootstrap CIs.

A prediction is an ordered path of candidate slots, shape (E, K'); the
ground truth is the target path, shape (E, K). Set metrics (precision,
recall, hit rate, NDCG, MRR) ignore the order of the targets; ``first_step``
checks that the first recommended concept is one the student actually
started next. When several targets were first attempted in the same time
window, ``first_group`` says how many leading targets count as "next".
"""

from __future__ import annotations

from math import comb

import numpy as np

METRICS = ("precision", "recall", "hit_rate", "ndcg", "mrr", "first_step")


def per_example(pred: np.ndarray, target: np.ndarray, first_group: np.ndarray | None = None) -> dict[str, np.ndarray]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    k_pred, k_true = pred.shape[1], target.shape[1]
    group = np.ones(len(target), dtype=int) if first_group is None else np.asarray(first_group)
    in_first_group = np.arange(k_true)[None, :] < group[:, None]
    hits = (pred[:, :, None] == target[:, None, :]).any(axis=2)  # (E, K')
    n_hits = hits.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k_pred + 2))
    ideal = discounts[: min(k_true, k_pred)].sum()
    first_rank = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
    return {
        "precision": n_hits / k_pred,
        "recall": n_hits / k_true,
        "hit_rate": (n_hits > 0).astype(float),
        "ndcg": (hits * discounts).sum(axis=1) / ideal,
        "mrr": np.where(first_rank > 0, 1.0 / np.maximum(first_rank, 1), 0.0),
        "first_step": ((pred[:, :1] == target) & in_first_group).any(axis=1).astype(float),
    }


def summarize(values: dict[str, np.ndarray], n_boot: int = 0, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean of each metric, plus a percentile bootstrap 95% CI when ``n_boot`` > 0."""
    out: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(seed)
    size = len(next(iter(values.values()))) if values else 0
    resamples = rng.integers(0, size, size=(n_boot, size)) if n_boot and size else None
    for name, v in values.items():
        entry = {"mean": float(np.mean(v)) if size else float("nan")}
        if resamples is not None:
            means = v[resamples].mean(axis=1)
            entry["ci_low"], entry["ci_high"] = (float(x) for x in np.percentile(means, [2.5, 97.5]))
        out[name] = entry
    return out


def chance(num_candidates: int, path_len: int) -> dict[str, float]:
    """Expected metrics for a uniformly random path of K distinct candidates.

    With N candidates of which K are targets, the number of hits in a random
    K-path is hypergeometric. Each rank holds a target with probability K/N,
    so precision, recall and NDCG all equal K/N.
    """
    n, k = num_candidates, path_len
    p_first_at = []
    miss_so_far = 1.0
    for rank in range(1, k + 1):
        p_first_at.append(miss_so_far * k / (n - rank + 1))
        miss_so_far *= (n - k - rank + 1) / (n - rank + 1)
    return {
        "precision": k / n,
        "recall": k / n,
        "hit_rate": 1.0 - comb(n - k, k) / comb(n, k),
        "ndcg": k / n,
        "mrr": sum(p / rank for rank, p in enumerate(p_first_at, start=1)),
        "first_step": 1.0 / n,
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    new_group = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    starts = np.flatnonzero(new_group)
    ends = np.r_[starts[1:], len(scores)] - 1
    ranks = np.empty(len(scores))
    ranks[order] = ((starts + ends) / 2.0 + 1.0)[np.cumsum(new_group) - 1]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_k(scores: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of the k highest scores per row, breaking ties at random."""
    tiebreak = rng.random(scores.shape)
    order = np.lexsort((tiebreak, -scores), axis=-1)
    return order[:, :k]
