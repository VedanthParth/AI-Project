"""Configuration for preprocessing, the model and training.

Each dataclass doubles as the CLI flag set for the command that uses it:
field ``path_len`` becomes ``--path-len``, with the field's default and help.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypeVar


def _opt(default: Any, help: str, cli_type: type | None = None) -> Any:
    meta = {"help": help}
    if cli_type is not None:
        meta["cli_type"] = cli_type
    return field(default=default, metadata=meta)


@dataclass
class DataConfig:
    """How raw interaction logs become (history, candidate set, target path) examples."""

    student_col: str = _opt("uuid", "student id column in the raw log")
    concept_col: str = _opt("ucid", "concept id column in the raw log")
    correct_col: str = _opt("is_correct", "correctness column (True/False or 1/0)")
    time_col: str = _opt("timestamp_TW", "ordering column (timestamp or numeric)")
    max_students: int | None = _opt(None, "subsample this many students (default: all)", int)
    path_len: int = _opt(3, "K, the number of concepts in a recommended path")
    num_candidates: int = _opt(20, "N, the size of each candidate set")
    min_history: int = _opt(3, "minimum interactions before the first target concept")
    windows: str = _opt("all", "'all' = one example per cut point, 'last' = paper's original last-K setup")
    max_windows_per_student: int = _opt(20, "cap on examples per student when windows='all'")
    negatives: str = _opt("uniform", "'uniform' or 'popularity' (train-split frequency) negative sampling")
    exclude_seen: bool = _opt(True, "never use concepts already in the history as negatives")
    split: tuple[float, float, float] = _opt((0.70, 0.15, 0.15), "train,val,test fractions by student", tuple)
    seed: int = _opt(42, "seed for subsampling, the split and candidate sampling")


@dataclass
class ModelConfig:
    """Pointer-network recommender (also sizes the GRU next-item baseline)."""

    embed_dim: int = _opt(32, "d, concept embedding size")
    hidden_dim: int = _opt(64, "H, GRU/LSTM hidden size")
    attn_heads: int = _opt(4, "heads in the candidate Transformer encoder")
    attn_layers: int = _opt(2, "Transformer layers over the candidate set (0 disables it)")
    ff_dim: int = _opt(128, "Transformer feed-forward size")
    dropout: float = _opt(0.1, "dropout on embeddings and inside the Transformer")
    kt_weight: float = _opt(0.5, "lambda_KT, weight of the auxiliary knowledge-tracing loss (0 disables it)")


@dataclass
class TrainConfig:
    """Supervised optimisation and evaluation settings."""

    lr: float = _opt(1e-3, "Adam learning rate")
    weight_decay: float = _opt(1e-5, "Adam weight decay")
    batch_size: int = _opt(64, "examples per batch")
    max_epochs: int = _opt(50, "upper bound on supervised epochs")
    patience: int = _opt(5, "stop after this many epochs without validation improvement")
    grad_clip: float = _opt(5.0, "gradient-norm clipping threshold")
    max_history: int = _opt(100, "keep only the last N interactions of each history")
    select_metric: str = _opt("ndcg", "validation metric used for early stopping")
    bootstrap: int = _opt(1000, "bootstrap resamples for final confidence intervals (0 = none)")
    seed: int = _opt(0, "seed for initialisation, shuffling and tie-breaking")
    device: str = _opt("auto", "'auto', 'cpu' or 'cuda'")


C = TypeVar("C")


def _str2bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(","))


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    group = parser.add_argument_group(cls.__name__)
    for f in dataclasses.fields(cls):
        kind = f.metadata.get("cli_type", type(f.default))
        flag = "--" + f.name.replace("_", "-")
        help_text = f"{f.metadata.get('help', '')} (default: {f.default})"
        if kind is bool:
            group.add_argument(flag, type=_str2bool, default=f.default, help=help_text)
        elif kind is tuple:
            group.add_argument(flag, type=_float_tuple, default=f.default, help=help_text)
        else:
            group.add_argument(flag, type=kind, default=f.default, help=help_text)


def dataclass_from_args(cls: type[C], args: argparse.Namespace) -> C:
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def dataclass_from_dict(cls: type[C], values: dict[str, Any]) -> C:
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items() if k in known}
    return cls(**kwargs)
