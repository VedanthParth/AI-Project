"""Walk-forward evaluation: train on the past, test on the following months.

The main split assigns whole students to train, validation and test at random,
so test students are unseen but their activity overlaps the training period
in time: popularity and Markov counts, for example, include months after a
test moment. A walk-forward fold with origin T removes that:

    train       every student's attempts before T - val_days
    validation  examples whose first target falls in [T - val_days, T), history up to T
    test        examples whose first target falls in [T, T + test_days), history up to T + test_days

Each student is cut at those dates into up to three copies (one per split),
so everything fitted on the training split (popularity, Markov counts, BKT,
the prerequisite graph, DKT, the models) sees only attempts before
T - val_days. A test example's history is the student's own past, which may
run past T. Test students are reported in two groups: new (no attempt before
T) and continuing (seen before T, possibly in training).

``prepare_fold`` writes an ordinary processed dataset, so ``grid``, ``train``
and the rest run on it unchanged; ``fold_report`` scores every run on it by
group.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from learnpath import metrics
from learnpath.config import DataConfig
from learnpath.preprocess import (
    Interactions,
    MarkovScorer,
    Processed,
    build_examples,
    concept_popularity,
    first_attempt_transitions,
    load_log,
    save_processed,
)

DAY = 86_400.0


def seconds(date: str) -> float:
    """Seconds since the epoch at 00:00 UTC on an ISO date."""
    return dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp()


def prepare_fold(
    inter: Interactions,
    out_dir: str | Path,
    cfg: DataConfig,
    origin: str,
    val_days: int = 30,
    test_days: int = 60,
) -> Processed:
    """A processed dataset for one walk-forward fold (see the module docstring)."""
    t0 = seconds(origin)
    val_start, test_end = t0 - val_days * DAY, t0 + test_days * DAY
    pieces, owners = [], []
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for s in range(inter.num_students):
        lo = int(inter.offsets[s])
        times = inter.times(s)
        for split, end, active_from in (("train", val_start, None), ("val", t0, val_start), ("test", test_end, t0)):
            n = int(np.searchsorted(times, end, side="left"))
            if n == 0 or (active_from is not None and times[n - 1] < active_from):
                continue
            splits[split].append(len(pieces))
            pieces.append(np.arange(lo, lo + n))
            owners.append(s)
    rows = np.concatenate(pieces)
    lengths = np.array([len(p) for p in pieces])
    fold = Interactions(
        student_uuid=inter.student_uuid[np.asarray(owners)],
        concept_names=inter.concept_names,
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        concept=inter.concept[rows],
        correct=inter.correct[rows],
        time=inter.time[rows],
    )
    ids = {name: np.asarray(v, dtype=np.int64) for name, v in splits.items()}
    popularity = concept_popularity(fold, ids["train"])
    markov = MarkovScorer(*first_attempt_transitions(fold, ids["train"]), popularity) if cfg.negatives == "markov" else None
    examples = {
        "train": build_examples(fold, ids["train"], popularity, cfg, seed=cfg.seed, markov=markov),
        "val": build_examples(fold, ids["val"], popularity, cfg, seed=cfg.seed + 1, markov=markov, first_target_from=val_start),
        "test": build_examples(fold, ids["test"], popularity, cfg, seed=cfg.seed + 2, markov=markov, first_target_from=t0),
    }
    test_students = np.unique(examples["test"].student)
    stats = {
        "origin": origin,
        "val_days": val_days,
        "test_days": test_days,
        "students": int(inter.num_students),
        "interactions": int(fold.num_interactions),
        "concepts": int(fold.num_concepts),
        **{f"{name}_students": int(len(np.unique(examples[name].student))) for name in ids},
        **{f"{name}_examples": int(len(examples[name])) for name in ids},
        "test_new_students": int(sum(fold.times(s)[0] >= t0 for s in test_students)),
    }
    proc = Processed(fold, ids, examples, popularity, cfg, stats)
    save_processed(proc, out_dir)
    return proc


def prepare_folds(
    log_path: str | Path,
    out_root: str | Path,
    cfg: DataConfig,
    origins: Sequence[str],
    val_days: int = 30,
    test_days: int = 60,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """One fold per origin, under ``out_root/fold-<origin>``, reading the log once."""
    inter = load_log(log_path, cfg)
    dirs = []
    for origin in origins:
        out = Path(out_root) / f"fold-{origin}"
        proc = prepare_fold(inter, out, cfg, origin, val_days, test_days)
        s = proc.stats
        log(
            f"fold {origin}: {s['train_examples']:,} train / {s['val_examples']:,} val / {s['test_examples']:,} test examples; "
            f"{s['test_new_students']:,} of {s['test_students']:,} test students are new"
        )
        dirs.append(out)
    return dirs


def new_students(proc: Processed, split: str = "test") -> np.ndarray:
    """Per example: whether its student had no attempt before the fold's origin."""
    t0 = seconds(proc.stats["origin"])
    first = proc.inter.time[proc.inter.offsets[:-1]]
    return first[proc.examples[split].student] >= t0


def fold_report(proc: Processed, runs_dir: str | Path, split: str = "test", n_boot: int = 1000) -> dict:
    """Metrics of every method under ``runs_dir`` on all, new and continuing students of the fold."""
    from learnpath.report import _collect_predictions

    ex = proc.examples[split]
    groups = {"all": np.ones(len(ex), bool), "new": new_students(proc, split)}
    groups["continuing"] = ~groups["new"]
    out: dict = {"origin": proc.stats["origin"], "examples": {g: int(m.sum()) for g, m in groups.items()}, "methods": {}}
    for name, seed_preds in _collect_predictions(Path(runs_dir), split).items():
        per_seed = [metrics.per_example(p, ex.target_slots, ex.first_group) for p in seed_preds]
        values = {k: np.mean([v[k] for v in per_seed], axis=0) for k in per_seed[0]}
        out["methods"][name] = {
            g: metrics.summarize({k: v[m] for k, v in values.items()}, n_boot, groups=ex.student[m]) if m.any() else None
            for g, m in groups.items()
        }
    return out
