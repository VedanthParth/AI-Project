"""Recommending from the whole catalogue, with no candidate list that holds the answer.

In the main experiments each example's 20 candidates are the student's 3 real
next concepts plus 17 others. A deployed recommender has no such list. Here a
shortlist generator ranks every concept the student has not started; its top
N become the candidate set, and a model orders them. Everything is scored
against the student's real next concepts, so a target the shortlist missed is
a miss for every method that uses it.

Shortlist generator: the Markov baseline's first-attempt transition scores
from the student's last few new concepts, fitted on training students, with
its small popularity term ordering everything else (``MarkovScorer``). Only
concepts the student has not attempted are eligible, as in the main task.
``prepare --negatives markov`` builds training candidate sets the same way,
so a model can learn to choose among the concepts a shortlist really offers.

Methods, each choosing 3 concepts from the same eligible set:

    popularity     the 3 unseen concepts most training students attempted
    markov         the generator's own top 3
    <run name>     a pointer-network checkpoint ordering the generator's top N
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from learnpath import metrics
from learnpath.batching import iterate_batches
from learnpath.preprocess import (
    Examples,
    Interactions,
    MarkovScorer,
    Processed,
    first_attempt_transitions,
    first_attempts,
)

NEG = -np.inf


class Shortlister:
    """Scores every concept for each example; concepts already in the history score -inf."""

    def __init__(self, markov: MarkovScorer, popularity: np.ndarray):
        self.markov = markov
        self.popularity = popularity

    @classmethod
    def fit(cls, proc: Processed) -> "Shortlister":
        transitions = first_attempt_transitions(proc.inter, proc.splits["train"])
        return cls(MarkovScorer(*transitions, proc.popularity), proc.popularity)

    def score_history(self, concepts: np.ndarray) -> np.ndarray:
        """Scores (C + 1,) for one history given as its attempted concepts, in order."""
        distinct, _ = first_attempts(concepts)
        scores = self.markov(distinct)
        scores[0] = NEG
        scores[distinct] = NEG
        return scores

    def rank(self, inter: Interactions, ex: Examples, n: int, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every example: the top-n shortlist, the k most popular eligible concepts, and how many are eligible.

        Works through the examples in chunks, so the (E, C + 1) score matrix is never held whole.
        """
        shortlist = np.empty((len(ex), n), np.int64)
        popular = np.empty((len(ex), k), np.int64)
        eligible = np.empty(len(ex), np.int64)
        for start in range(0, len(ex), chunk):
            rows = range(start, min(start + chunk, len(ex)))
            scores = np.stack([self.score_history(inter.sequence(ex.student[i])[0][: ex.cut[i]]) for i in rows])
            shortlist[rows.start : rows.stop] = top_concepts(scores, n)
            popular[rows.start : rows.stop] = top_concepts(np.where(np.isfinite(scores), self.popularity, NEG), k)
            eligible[rows.start : rows.stop] = np.isfinite(scores).sum(axis=1)
        return shortlist, popular, eligible


def top_concepts(scores: np.ndarray, n: int) -> np.ndarray:
    """The n highest-scoring concept ids per row, best first."""
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def shortlist_examples(ex: Examples, shortlist: np.ndarray) -> tuple[Examples, np.ndarray]:
    """``ex`` with the shortlist as its candidates, and which targets the shortlist contains (E, K).

    A target the shortlist missed gets slot 0; only teacher forcing reads target
    slots, and nothing here trains.
    """
    found = shortlist[:, None, :] == ex.targets[:, :, None]  # (E, K, N)
    slots = np.where(found.any(axis=2), found.argmax(axis=2), 0).astype(np.int64)
    fields = {**dataclasses.asdict(ex), "candidates": shortlist.astype(np.int32), "target_slots": slots}
    return Examples(**fields), found.any(axis=2)


@torch.no_grad()
def pointer_paths(model, inter: Interactions, ex: Examples, k: int, max_history: int, device, batch_size: int = 256) -> np.ndarray:
    """Greedy pointer-network paths as concept ids, shape (E, K)."""
    model.eval()
    slots = [
        model.recommend(batch.to(device), k).actions.cpu().numpy()
        for batch in iterate_batches(inter, ex, batch_size, max_history)
    ]
    return np.take_along_axis(ex.candidates, np.concatenate(slots), axis=1)


def evaluate_catalogue(
    proc: Processed,
    split: str,
    checkpoints: dict[str, Path],
    *,
    shortlist_size: int = 20,
    max_examples: int | None = None,
    n_boot: int = 1000,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict:
    from learnpath.train import load_checkpoint
    from learnpath.utils import get_device

    ex = proc.examples[split]
    if max_examples is not None and max_examples < len(ex):
        rows = np.sort(np.random.default_rng(0).choice(len(ex), max_examples, replace=False))
        ex = Examples(**{f.name: getattr(ex, f.name)[rows] for f in dataclasses.fields(Examples)})
    k = ex.targets.shape[1]
    log(f"scoring all {proc.inter.num_concepts:,} concepts for {len(ex):,} {split} examples")
    shortlist, popular, eligible = Shortlister.fit(proc).rank(proc.inter, ex, shortlist_size, k)
    short_ex, found = shortlist_examples(ex, shortlist)
    paths = {"popularity": popular, "markov": shortlist[:, :k]}
    dev = get_device(device)
    for name, path in checkpoints.items():
        model, ckpt = load_checkpoint(path, dev)
        paths[name] = pointer_paths(model, proc.inter, short_ex, k, ckpt["train_config"]["max_history"], dev)
        log(f"  {name}: done")

    result = {
        "split": split,
        "examples": int(len(ex)),
        "eligible_concepts": float(eligible.mean()),
        "shortlist_size": shortlist_size,
        "shortlist": {
            "target_recall": float(found.mean()),
            "all_targets": float(found.all(axis=1).mean()),
            "any_target": float(found.any(axis=1).mean()),
        },
        "methods": {
            name: metrics.summarize(metrics.per_example(p, ex.targets, ex.first_group), n_boot, groups=ex.student)
            for name, p in paths.items()
        },
    }
    return result
