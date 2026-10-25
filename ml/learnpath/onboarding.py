"""A new student with no history: a short questionnaire becomes a history the models can read.

The questionnaire (the demo's "New student" page) asks for:

1. the school stage, elementary or junior high;
2. for each strand of that stage (e.g. *Algebra*), strong, okay or weak;
3. for strands rated okay or weak, which of their units (e.g. *Quadratic
   equations*) the student is good or bad at.

``checklist_history`` turns the answers into the attempt history every model
in this package reads. A unit marked good contributes its most-practised
exercises, each answered correctly ``attempts`` times; a unit marked bad
contributes the same exercises answered mostly wrong. A strong strand marks
all its units good. The attempts follow the curriculum order
(``curriculum_position``: how far into their own sequence training students
typically start each exercise), so the history reads like a student who
worked through those topics in the usual order.

Self-reports are not attempts, so how well this works has to be measured,
not assumed. ``simulate`` fills the questionnaire in from held-out students'
real histories (the units they practised most recently, good or bad by their
accuracy there, with an optional share of answers flipped as a student
misjudging themself would), recommends from that history alone, and scores
the result against what each student really did next. It compares the same
models given the student's full real history, and popularity, which is all a
system knows about a student it has never seen.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from learnpath import metrics
from learnpath.catalogue import Shortlister, pointer_paths, shortlist_examples
from learnpath.preprocess import Examples, Interactions, Processed, first_attempts

GOOD, BAD = "good", "bad"
WRONG_PATTERN = (False, True, False)  # a unit marked bad: mostly wrong, one success
TOPICS_CSV = Path(__file__).resolve().parents[1] / "resources" / "junyi_topics_en.csv"


@dataclass
class Topics:
    """Junyi's strands and units, with each unit's exercises as concept ids."""

    strands: pd.DataFrame  # id, stage, name_en
    units: pd.DataFrame  # id, parent (strand id), stage, name_en, concepts (np.ndarray of ids)
    concept_unit: np.ndarray  # (C + 1,) row of ``units`` for each concept, -1 when unknown

    @classmethod
    def load(cls, concept_names: np.ndarray, content_csv: str | Path, topics_csv: str | Path = TOPICS_CSV) -> "Topics":
        table = pd.read_csv(topics_csv)
        units = table[table.kind == "unit"].reset_index(drop=True)
        info = pd.read_csv(content_csv).set_index("ucid").reindex(concept_names[1:])
        row_of = {uid: i for i, uid in enumerate(units.id)}
        concept_unit = np.full(len(concept_names), -1, dtype=np.int64)
        concept_unit[1:] = [row_of.get(u, -1) for u in info["level3_id"].fillna("")]
        units["concepts"] = [np.flatnonzero(concept_unit == i) for i in range(len(units))]
        return cls(table[table.kind == "strand"].reset_index(drop=True), units, concept_unit)

    def representatives(self, unit: int, popularity: np.ndarray, per_unit: int) -> np.ndarray:
        """The unit's ``per_unit`` exercises most training students attempted."""
        concepts = self.units.concepts[unit]
        return concepts[np.argsort(-popularity[concepts], kind="stable")[:per_unit]]


def curriculum_position(inter: Interactions, students: np.ndarray) -> np.ndarray:
    """(C + 1,) mean relative position (0 = first, 1 = last) at which students start each concept."""
    total = np.zeros(inter.num_concepts + 1)
    count = np.zeros(inter.num_concepts + 1)
    for s in students:
        distinct, _ = first_attempts(inter.sequence(s)[0])
        if len(distinct) > 1:
            total[distinct] += np.arange(len(distinct)) / (len(distinct) - 1)
            count[distinct] += 1
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return np.where(np.isnan(mean), np.nanmean(mean[1:]) if np.isfinite(mean[1:]).any() else 0.5, mean)


def checklist_history(
    marks: dict[int, str],
    topics: Topics,
    popularity: np.ndarray,
    position: np.ndarray,
    per_unit: int = 4,
    attempts: int = 3,
    current: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """(concepts, correct) attempt arrays from the questionnaire's answers, in curriculum order.

    ``marks`` maps unit rows to "good" or "bad". ``current`` lists the exercises
    the student says they finished last, oldest first. The exercises before the
    last of them in its own unit are taken as done (one correct attempt each),
    the way a placement test assumes the earlier steps of a sequence; the
    current exercises end the history, so the models read them as the
    student's latest work.
    """
    rows = []
    for unit, mark in marks.items():
        if mark not in (GOOD, BAD):
            raise ValueError(f"unit marks must be {GOOD!r} or {BAD!r}, got {mark!r}")
        for c in topics.representatives(unit, popularity, per_unit):
            rows.append((position[c], int(c), mark, attempts))
    current = [int(c) for c in current]
    if current:
        unit = topics.concept_unit[current[-1]]
        if unit >= 0:
            before = [c for c in topics.units.concepts[unit].tolist() if position[c] < position[current[-1]] and c not in current]
            rows = [r for r in rows if r[1] not in before] + [(position[c], c, GOOD, 1) for c in before]
        rows = [r for r in rows if r[1] not in current]
    concepts, correct = [], []
    for _, c, mark, n in sorted(rows):
        concepts += [c] * n
        correct += [True] * n if mark == GOOD else [WRONG_PATTERN[i % len(WRONG_PATTERN)] for i in range(n)]
    for c in current:
        concepts += [c] * attempts
        correct += [True] * attempts
    return np.asarray(concepts, dtype=np.int32), np.asarray(correct, dtype=np.int8)


def marks_from_history(
    concepts: np.ndarray, correct: np.ndarray, topics: Topics, max_units: int, good_accuracy: float
) -> dict[int, str]:
    """The questionnaire a student would fill in: their most recently practised units, good or bad by accuracy."""
    units = topics.concept_unit[concepts]
    keep = units >= 0
    units, right = units[keep], correct[keep].astype(float)
    if len(units) == 0:
        return {}
    last_seen = {u: i for i, u in enumerate(units.tolist())}
    recent = sorted(last_seen, key=last_seen.get, reverse=True)[:max_units]
    return {u: GOOD if right[units == u].mean() >= good_accuracy else BAD for u in recent}


def histories_as_examples(
    histories: list[tuple[np.ndarray, np.ndarray]], like: Examples
) -> tuple[Interactions, Examples]:
    """Synthetic students (one per history) and examples whose history is all of it, targets copied from ``like``."""
    lengths = np.array([len(c) for c, _ in histories], dtype=np.int64)
    inter = Interactions(
        student_uuid=np.array([f"questionnaire-{i}" for i in range(len(histories))]),
        concept_names=np.array([]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        concept=np.concatenate([c for c, _ in histories]).astype(np.int32) if len(histories) else np.zeros(0, np.int32),
        correct=np.concatenate([y for _, y in histories]).astype(np.int8) if len(histories) else np.zeros(0, np.int8),
        time=np.zeros(int(lengths.sum())),
    )
    ex = dataclasses.replace(like, student=np.arange(len(histories), dtype=np.int32), cut=lengths.astype(np.int32))
    return inter, ex


def simulate(
    proc: Processed,
    topics: Topics,
    split: str,
    checkpoints: dict[str, Path],
    *,
    max_units: int = 6,
    noise: float = 0.0,
    current: int = 0,
    good_accuracy: float = 0.7,
    per_unit: int = 4,
    shortlist_size: int = 50,
    max_examples: int | None = 5000,
    n_boot: int = 1000,
    seed: int = 0,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict:
    """Score recommendations made from a simulated questionnaire against held-out students' real next concepts."""
    from learnpath.train import load_checkpoint
    from learnpath.utils import get_device

    rng = np.random.default_rng(seed)
    ex = proc.examples[split]
    if max_examples is not None and max_examples < len(ex):
        rows = np.sort(rng.choice(len(ex), max_examples, replace=False))
        ex = Examples(**{f.name: getattr(ex, f.name)[rows] for f in dataclasses.fields(Examples)})
    k = ex.targets.shape[1]
    position = curriculum_position(proc.inter, proc.splits["train"])

    histories, questions = [], []
    for s, cut in zip(ex.student.tolist(), ex.cut.tolist()):
        concepts, correct = proc.inter.sequence(s)
        marks = marks_from_history(concepts[:cut], correct[:cut], topics, max_units, good_accuracy)
        flip = rng.random(len(marks)) < noise
        marks = {u: (BAD if m == GOOD else GOOD) if f else m for (u, m), f in zip(marks.items(), flip)}
        named = first_attempts(concepts[:cut])[0][-current:] if current else ()
        histories.append(checklist_history(marks, topics, proc.popularity, position, per_unit, current=named))
        questions.append(len(marks))
    has_answers = np.array([len(c) > 0 for c, _ in histories])
    log(f"{has_answers.mean():.1%} of {len(ex):,} {split} examples have a practised unit to report; scoring those")
    keep = np.flatnonzero(has_answers)
    ex = Examples(**{f.name: getattr(ex, f.name)[keep] for f in dataclasses.fields(Examples)})
    histories = [histories[i] for i in keep]
    q_inter, q_ex = histories_as_examples(histories, ex)

    shortlister = Shortlister.fit(proc)
    real_short, popular, _ = shortlister.rank(proc.inter, ex, shortlist_size, k)
    q_short, _, _ = shortlister.rank(q_inter, q_ex, shortlist_size, k)
    real_short_ex, _ = shortlist_examples(ex, real_short)
    q_short_ex, q_found = shortlist_examples(q_ex, q_short)
    new_student_popular = np.argsort(-proc.popularity[1:], kind="stable")[:k] + 1

    paths = {
        "popularity (knows nothing)": np.tile(new_student_popular, (len(ex), 1)),
        "markov, full history": real_short[:, :k],
        "markov, questionnaire": q_short[:, :k],
    }
    dev = get_device(device)
    for name, path in checkpoints.items():
        model, ckpt = load_checkpoint(path, dev)
        max_history = ckpt["train_config"]["max_history"]
        paths[f"{name}, full history"] = pointer_paths(model, proc.inter, real_short_ex, k, max_history, dev)
        paths[f"{name}, questionnaire"] = pointer_paths(model, q_inter, q_short_ex, k, max_history, dev)
    return {
        "split": split,
        "examples": int(len(ex)),
        "settings": {
            "max_units": max_units, "noise": noise, "current": current, "good_accuracy": good_accuracy,
            "per_unit": per_unit, "shortlist_size": shortlist_size,
        },
        "questions_answered": float(np.mean([questions[i] for i in keep])),
        "questionnaire_shortlist_recall": float(q_found.mean()),
        "methods": {
            name: metrics.summarize(metrics.per_example(p, ex.targets, ex.first_group), n_boot, groups=ex.student)
            for name, p in paths.items()
        },
    }
