"""Turn example indices into padded tensors for the models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from pathtrace.preprocess import Examples, Interactions


@dataclass
class Batch:
    hist_concepts: torch.Tensor  # (B, T) long, right-padded with 0
    hist_correct: torch.Tensor  # (B, T) float
    lengths: torch.Tensor  # (B,) long, kept on CPU for sequence packing
    candidates: torch.Tensor  # (B, N) long concept ids
    target_slots: torch.Tensor  # (B, K) long positions in ``candidates``
    targets: torch.Tensor  # (B, K) long concept ids
    index: np.ndarray  # (B,) example indices into the split

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            hist_concepts=self.hist_concepts.to(device),
            hist_correct=self.hist_correct.to(device),
            lengths=self.lengths,
            candidates=self.candidates.to(device),
            target_slots=self.target_slots.to(device),
            targets=self.targets.to(device),
            index=self.index,
        )


def make_batch(inter: Interactions, ex: Examples, index: np.ndarray, max_history: int) -> Batch:
    lengths = np.minimum(ex.cut[index], max_history)
    width = int(lengths.max())
    concepts = np.zeros((len(index), width), dtype=np.int64)
    correct = np.zeros((len(index), width), dtype=np.float32)
    for row, (i, length) in enumerate(zip(index, lengths)):
        end = inter.offsets[ex.student[i]] + ex.cut[i]
        concepts[row, :length] = inter.concept[end - length : end]
        correct[row, :length] = inter.correct[end - length : end]
    return Batch(
        hist_concepts=torch.from_numpy(concepts),
        hist_correct=torch.from_numpy(correct),
        lengths=torch.from_numpy(lengths.astype(np.int64)),
        candidates=torch.from_numpy(ex.candidates[index].astype(np.int64)),
        target_slots=torch.from_numpy(ex.target_slots[index].astype(np.int64)),
        targets=torch.from_numpy(ex.targets[index].astype(np.int64)),
        index=np.asarray(index),
    )


def iterate_batches(
    inter: Interactions,
    ex: Examples,
    batch_size: int,
    max_history: int,
    rng: np.random.Generator | None = None,
) -> Iterator[Batch]:
    """Yield batches in order, or shuffled when ``rng`` is given."""
    order = rng.permutation(len(ex)) if rng is not None else np.arange(len(ex))
    for start in range(0, len(order), batch_size):
        yield make_batch(inter, ex, order[start : start + batch_size], max_history)
