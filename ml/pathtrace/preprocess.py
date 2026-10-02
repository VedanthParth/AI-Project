"""Raw interaction log -> student split -> path-recommendation examples.

Pipeline (``prepare``):

1. ``load_log`` reads only the four needed columns with pyarrow (dictionary-
   encoded ids, so the full 16M-row Junyi log fits in memory), optionally
   subsamples students, and sorts each student's attempts by time.
2. ``split_students`` assigns whole students to train/val/test, so no student
   appears in more than one split.
3. ``concept_popularity`` counts, per concept, how many *training* students
   attempted it. Popularity-weighted sampling and the Popularity baseline use
   only this, never validation or test data.
4. ``build_examples`` turns each student's log into sliding-window examples.
   For the student's distinct concepts in first-attempt order d_0, d_1, ...,
   a cut at d_j gives: history = every attempt before the first attempt at
   d_j; target path = (d_j, ..., d_{j+K-1}), the next K new concepts; and
   a candidate set = the targets plus N-K sampled negatives, shuffled.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from pathtrace.config import DataConfig, dataclass_from_dict
from pathtrace.utils import load_json, save_json

SPLITS = ("train", "val", "test")
PAD = "<pad>"


@dataclass
class Interactions:
    """All students' attempts, concatenated and sorted by (student, time).

    Student ``s`` owns rows ``offsets[s]:offsets[s + 1]``. Concept ids run
    1..C; id 0 is reserved for padding.
    """

    student_uuid: np.ndarray  # (S,) raw student ids
    concept_names: np.ndarray  # (C + 1,) raw concept ids; index 0 is PAD
    offsets: np.ndarray  # (S + 1,) int64
    concept: np.ndarray  # (M,) int32
    correct: np.ndarray  # (M,) int8

    @property
    def num_students(self) -> int:
        return len(self.student_uuid)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_names) - 1

    @property
    def num_interactions(self) -> int:
        return len(self.concept)

    def sequence(self, student: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.offsets[student], self.offsets[student + 1]
        return self.concept[lo:hi], self.correct[lo:hi]


@dataclass
class Examples:
    """One row per (student, cut point). Concept ids index ``concept_names``."""

    student: np.ndarray  # (E,) int32
    cut: np.ndarray  # (E,) int32: history = the student's first ``cut`` attempts
    targets: np.ndarray  # (E, K) int32, in first-attempt order
    candidates: np.ndarray  # (E, N) int32, shuffled
    target_slots: np.ndarray  # (E, K) int64: candidates[e, target_slots[e, k]] == targets[e, k]

    def __len__(self) -> int:
        return len(self.student)


@dataclass
class Processed:
    inter: Interactions
    splits: dict[str, np.ndarray]
    examples: dict[str, Examples]
    popularity: np.ndarray  # (C + 1,) training students per concept
    data_cfg: DataConfig
    stats: dict


# --------------------------------------------------------------------------- loading


def _parse_correct(values: pd.Series) -> np.ndarray:
    lowered = values.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    if mapped.isna().any():
        bad = sorted(lowered[mapped.isna()].unique())[:5]
        raise ValueError(f"unrecognised correctness values: {bad}")
    return mapped.to_numpy(np.int8)


def _parse_time(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.to_numpy(np.float64)
    text = values.astype(str).str.replace(r"\s*UTC$", "", regex=True)
    parsed = pd.to_datetime(text, format="ISO8601", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(text, format="mixed", errors="coerce")
    if parsed.isna().any():
        bad = text[parsed.isna()].head(3).tolist()
        raise ValueError(f"could not parse {int(parsed.isna().sum())} timestamps, e.g. {bad}")
    return parsed.to_numpy("datetime64[ns]").astype(np.int64).astype(np.float64)


def _dense_ids(codes: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-index codes to 0..U-1 so that ids follow the sorted raw names."""
    used, inverse = np.unique(codes, return_inverse=True)
    used_names = names[used]
    order = np.argsort(used_names, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse], used_names[order]


def load_log(path: str | Path, cfg: DataConfig) -> Interactions:
    cols = [cfg.student_col, cfg.concept_col, cfg.correct_col, cfg.time_col]
    as_dict = pa.dictionary(pa.int32(), pa.string())
    convert = pacsv.ConvertOptions(
        include_columns=cols,
        column_types={
            cfg.student_col: as_dict,
            cfg.concept_col: as_dict,
            cfg.correct_col: pa.string(),
            cfg.time_col: pa.string(),
        },
    )
    table = pacsv.read_csv(path, convert_options=convert)
    df = table.to_pandas()
    df = df.dropna(subset=cols).reset_index(drop=True)

    student_codes = df[cfg.student_col].cat.codes.to_numpy()
    student_names = np.asarray(df[cfg.student_col].cat.categories, dtype=str)
    if cfg.max_students is not None:
        observed = np.unique(student_codes)
        rng = np.random.default_rng(cfg.seed)
        keep = rng.choice(observed, size=min(cfg.max_students, len(observed)), replace=False)
        mask = np.isin(student_codes, keep)
        df = df.loc[mask].reset_index(drop=True)
        student_codes = student_codes[mask]

    student, student_uuid = _dense_ids(student_codes, student_names)
    concept, concept_names = _dense_ids(
        df[cfg.concept_col].cat.codes.to_numpy(), np.asarray(df[cfg.concept_col].cat.categories, dtype=str)
    )
    correct = _parse_correct(df[cfg.correct_col])
    time = _parse_time(df[cfg.time_col])

    # Primary key: student; then time; ties keep file order.
    order = np.lexsort((np.arange(len(df)), time, student))
    counts = np.bincount(student, minlength=len(student_uuid))
    return Interactions(
        student_uuid=student_uuid,
        concept_names=np.concatenate([[PAD], concept_names]),
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        concept=(concept[order] + 1).astype(np.int32),
        correct=correct[order],
    )


# --------------------------------------------------------------------------- splits


def split_students(num_students: int, fractions: tuple[float, float, float], seed: int) -> dict[str, np.ndarray]:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must be three numbers summing to 1, got {fractions}")
    perm = np.random.default_rng(seed).permutation(num_students)
    n_train = int(round(fractions[0] * num_students))
    n_val = int(round(fractions[1] * num_students))
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }


def concept_popularity(inter: Interactions, students: np.ndarray) -> np.ndarray:
    """Number of the given students who attempted each concept, shape (C + 1,)."""
    owner = np.repeat(np.arange(inter.num_students), np.diff(inter.offsets))
    selected = np.isin(owner, students)
    width = inter.num_concepts + 1
    pairs = np.unique(owner[selected].astype(np.int64) * width + inter.concept[selected])
    return np.bincount(pairs % width, minlength=width).astype(np.float64)


def first_attempts(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct concepts in first-attempt order, and the index of each first attempt."""
    unique, first = np.unique(sequence, return_index=True)
    order = np.argsort(first, kind="stable")
    return unique[order], first[order]


# --------------------------------------------------------------------------- examples


def _sample_negatives(
    rng: np.random.Generator, cdf: np.ndarray, n: int, excluded: np.ndarray, targets: np.ndarray
) -> np.ndarray | None:
    num_concepts = len(cdf)
    blocked = set(excluded.tolist())
    if num_concepts - len(blocked) < n:
        # Small vocabularies only: fall back to allowing already-seen concepts.
        blocked = set(targets.tolist())
        if num_concepts - len(blocked) < n:
            return None
    chosen: list[int] = []
    while len(chosen) < n:
        draws = np.minimum(np.searchsorted(cdf, rng.random(2 * n + 8), side="right"), num_concepts - 1) + 1
        for concept in draws.tolist():
            if concept not in blocked:
                blocked.add(concept)
                chosen.append(concept)
                if len(chosen) == n:
                    break
    return np.asarray(chosen, dtype=np.int32)


def build_examples(
    inter: Interactions, students: np.ndarray, popularity: np.ndarray, cfg: DataConfig, seed: int
) -> Examples:
    k, n = cfg.path_len, cfg.num_candidates
    if n > inter.num_concepts:
        raise ValueError(f"num_candidates={n} exceeds the {inter.num_concepts} concepts in the data")
    if cfg.windows not in {"all", "last"}:
        raise ValueError(f"windows must be 'all' or 'last', got {cfg.windows!r}")
    if cfg.negatives == "uniform":
        weights = np.ones(inter.num_concepts)
    elif cfg.negatives == "popularity":
        weights = popularity[1:] + 1.0
    else:
        raise ValueError(f"negatives must be 'uniform' or 'popularity', got {cfg.negatives!r}")
    cdf = np.cumsum(weights / weights.sum())
    rng = np.random.default_rng(seed)

    rows_student, rows_cut, rows_targets, rows_cands, rows_slots = [], [], [], [], []
    for s in students:
        sequence, _ = inter.sequence(s)
        distinct, firsts = first_attempts(sequence)
        cuts = np.arange(1, len(distinct) - k + 1)
        cuts = cuts[firsts[cuts] >= cfg.min_history]
        if cuts.size == 0:
            continue
        if cfg.windows == "last":
            cuts = cuts[-1:]
        elif cuts.size > cfg.max_windows_per_student:
            cuts = np.sort(rng.choice(cuts, cfg.max_windows_per_student, replace=False))
        for j in cuts:
            targets = distinct[j : j + k]
            # distinct[:j] is exactly the set of concepts seen in the history.
            excluded = distinct[: j + k] if cfg.exclude_seen else targets
            negatives = _sample_negatives(rng, cdf, n - k, excluded, targets)
            if negatives is None:
                continue
            perm = rng.permutation(n)
            rows_student.append(s)
            rows_cut.append(firsts[j])
            rows_targets.append(targets)
            rows_cands.append(np.concatenate([targets, negatives])[perm])
            rows_slots.append(np.argsort(perm)[:k])

    if not rows_student:
        return Examples(
            student=np.zeros(0, np.int32),
            cut=np.zeros(0, np.int32),
            targets=np.zeros((0, k), np.int32),
            candidates=np.zeros((0, n), np.int32),
            target_slots=np.zeros((0, k), np.int64),
        )
    return Examples(
        student=np.asarray(rows_student, np.int32),
        cut=np.asarray(rows_cut, np.int32),
        targets=np.stack(rows_targets).astype(np.int32),
        candidates=np.stack(rows_cands).astype(np.int32),
        target_slots=np.stack(rows_slots).astype(np.int64),
    )


# --------------------------------------------------------------------------- persistence


def prepare(log_path: str | Path, out_dir: str | Path, cfg: DataConfig) -> Processed:
    inter = load_log(log_path, cfg)
    splits = split_students(inter.num_students, cfg.split, cfg.seed)
    popularity = concept_popularity(inter, splits["train"])
    examples = {
        name: build_examples(inter, splits[name], popularity, cfg, seed=cfg.seed + i)
        for i, name in enumerate(SPLITS)
    }
    stats = {
        "students": inter.num_students,
        "interactions": inter.num_interactions,
        "concepts": inter.num_concepts,
        "correct_rate": float(inter.correct.mean()) if inter.num_interactions else 0.0,
        **{f"{name}_students": len(splits[name]) for name in SPLITS},
        **{f"{name}_examples": len(examples[name]) for name in SPLITS},
    }
    processed = Processed(inter, splits, examples, popularity, cfg, stats)
    save_processed(processed, out_dir)
    return processed


def save_processed(p: Processed, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "interactions.npz",
        student_uuid=p.inter.student_uuid,
        concept_names=p.inter.concept_names,
        offsets=p.inter.offsets,
        concept=p.inter.concept,
        correct=p.inter.correct,
        popularity=p.popularity,
        **{f"split_{name}": p.splits[name] for name in SPLITS},
    )
    for name in SPLITS:
        np.savez_compressed(out / f"examples_{name}.npz", **dataclasses.asdict(p.examples[name]))
    save_json({"data_config": dataclasses.asdict(p.data_cfg), "stats": p.stats}, out / "meta.json")


def load_processed(data_dir: str | Path) -> Processed:
    data = Path(data_dir)
    meta = load_json(data / "meta.json")
    with np.load(data / "interactions.npz") as z:
        inter = Interactions(
            student_uuid=z["student_uuid"],
            concept_names=z["concept_names"],
            offsets=z["offsets"],
            concept=z["concept"],
            correct=z["correct"],
        )
        popularity = z["popularity"]
        splits = {name: z[f"split_{name}"] for name in SPLITS}
    examples = {}
    for name in SPLITS:
        with np.load(data / f"examples_{name}.npz") as z:
            examples[name] = Examples(**{f.name: z[f.name] for f in dataclasses.fields(Examples)})
    return Processed(
        inter=inter,
        splits=splits,
        examples=examples,
        popularity=popularity,
        data_cfg=dataclass_from_dict(DataConfig, meta["data_config"]),
        stats=meta["stats"],
    )
