"""Plain-language explanations of a recommendation, for the app's charts and reasons.

Nothing here is learned: every number comes from what the models already produce.

- ``topic_factors``: five scores between 0 and 1 for one topic, the axes of the app's
  "why this topic" spider chart:
    ready    share of the topic's building blocks (inferred prerequisites) already done
    room     room to grow: 1 - the estimated chance the student already knows it (BKT)
    recent   how strongly it follows the student's latest topics (Markov score, as a share
             of the strongest option's)
    level    how close it sits in the syllabus to what the student did last
    gain     the expected improvement from practising it (BKT gain, ranked among the options)
  plus ``sure``, the recommender's probability for it at its step (None for non-model paths),
  which the app shows only in its technical panel.
- ``path_profile``: the same idea for a three-topic path, for comparing ways of choosing.
- ``topic_reason``: one or two plain sentences on why a topic is in a plan.
- ``area_strengths``: estimated understanding per area of maths, for the strengths chart.

Rankings are percentiles among the topics the student could be offered, so 1 means "the
strongest of all the options" and 0 "the weakest".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

CONFIDENT = 0.8  # estimated understanding above which a topic counts as known well
NEW_TO_YOU = 0.4  # and below which it counts as new


def percentile(values: np.ndarray) -> np.ndarray:
    """Each value's rank among all of them, from 0 (lowest) to 1 (highest); ties share their average rank."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.ones(len(values))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    return ((start + (counts - 1) / 2) / (len(values) - 1))[inverse.ravel()]


def ranked(values: np.ndarray, among: np.ndarray) -> np.ndarray:
    """A full-size array holding the percentile of ``values[among]`` at those positions, 0 elsewhere."""
    out = np.zeros(len(values))
    out[among] = percentile(values[among])
    return out


def unmet(concept: int, parents: Sequence[Sequence[int]], done: set[int]) -> list[int]:
    return [p for p in parents[concept] if p not in done]


def scaled(values: np.ndarray, among: np.ndarray) -> np.ndarray:
    """A full-size array holding ``values[among]`` as a share of their maximum, 0 elsewhere."""
    out = np.zeros(len(values))
    top = float(np.max(values[among])) if len(among) else 0.0
    if top > 0:
        out[among] = np.clip(values[among] / top, 0.0, 1.0)
    return out


def syllabus_level(position: np.ndarray, started: Sequence[int], width: float = 0.25) -> np.ndarray:
    """1 for topics at the syllabus position of the student's last three topics, falling to 0 at ``width`` away."""
    if len(started) == 0:
        return np.ones(len(position))
    now = float(np.mean(position[list(started)[-3:]]))
    return np.clip(1 - np.abs(position - now) / width, 0.0, 1.0)


def readiness(concept: int, parents: Sequence[Sequence[int]], done: set[int]) -> float:
    blocks = parents[concept]
    return 1.0 if not blocks else float(np.mean([p in done for p in blocks]))


def topic_factors(
    concept: int,
    *,
    done: set[int],
    parents: Sequence[Sequence[int]],
    mastery: np.ndarray,
    recent: np.ndarray,
    level: np.ndarray,
    gain: np.ndarray,
    sure: float | None = None,
) -> dict[str, float | None]:
    """The five 0-1 scores for one topic (and ``sure``); ``done`` includes earlier steps of the same path."""
    return {
        "ready": round(readiness(concept, parents, done), 3),
        "room": round(float(1 - mastery[concept]), 3),
        "recent": round(float(recent[concept]), 3),
        "level": round(float(level[concept]), 3),
        "gain": round(float(gain[concept]), 3),
        "sure": None if sure is None else round(float(sure), 3),
    }


def path_profile(
    concepts: Sequence[int],
    *,
    history: set[int],
    parents: Sequence[Sequence[int]],
    mastery: np.ndarray,
    recent: np.ndarray,
    level: np.ndarray,
    expected_gain: np.ndarray,
    best_gain: float,
    progress: float | None = None,
    matched: float | None = None,
) -> dict[str, float]:
    """A path's scores between 0 and 1, comparable across ways of choosing for the same student.

    ``expected_gain`` is the raw expected improvement per concept; the path's total is shown
    as a share of ``best_gain``, the largest total among the paths being compared (as
    ``progress`` is, by the caller).
    """
    done, skips = set(history), 0
    for c in concepts:
        skips += bool(unmet(c, parents, done))
        done.add(c)
    profile = {
        "ready": 1 - skips / len(concepts),
        "room": float(np.mean([1 - mastery[c] for c in concepts])),
        "recent": float(np.mean([recent[c] for c in concepts])),
        "level": float(np.mean([level[c] for c in concepts])),
        "gain": float(sum(expected_gain[c] for c in concepts) / best_gain) if best_gain > 0 else 0.0,
    }
    if progress is not None:
        profile["progress"] = progress
    if matched is not None:
        profile["matched"] = matched
    return {k: round(min(max(v, 0.0), 1.0), 3) for k, v in profile.items()}


def topic_reason(
    concept: int,
    *,
    history: set[int],
    earlier: Sequence[int],
    parents: Sequence[Sequence[int]],
    mastery: np.ndarray,
    recent: np.ndarray,
    name: Callable[[int], str],
    needs_help: bool = False,
    chapter: str = "",
    person: str = "you",
) -> str:
    """One or two plain sentences on why ``concept`` is in the plan, from the same facts as its factors.

    ``person`` is "you" when talking to the student, "they" when describing an example student.
    """
    sub, obj, own = ("you", "you", "your") if person == "you" else ("they", "them", "their")
    blocks = parents[concept]
    missing = [p for p in blocks if p not in history and p not in earlier]
    covered_earlier = [p for p in blocks if p in earlier]
    known = sorted((p for p in blocks if p in history), key=lambda p: -mastery[p])
    sentences = []
    if missing:
        sentences.append(f"Its building block “{name(missing[0])}” isn't done yet, so it is worth a look first.")
    elif covered_earlier:
        sentences.append(f"It comes after “{name(covered_earlier[0])}”, earlier in this plan.")
    elif known:
        how = f"which {sub} know well" if mastery[known[0]] >= CONFIDENT else f"which {sub} have already practised"
        sentences.append(f"It builds on “{name(known[0])}”, {how}.")
    if needs_help and chapter:
        sentences.append(f"It is part of {chapter}, which {sub} said needs help.")
    elif recent[concept] >= 0.9:
        sentences.append(f"Students usually take it up right after the topics {sub} did last.")
    if len(sentences) < 2 and not missing:  # a missing building block outweighs "a quick win"
        if mastery[concept] < NEW_TO_YOU:
            sentences.append(f"It is new to {obj}, so there is a lot to gain.")
        elif mastery[concept] >= CONFIDENT:
            sentences.append(f"{sub.capitalize()} may know much of it already, so it should be a quick win.")
    if not sentences:
        sentences.append(f"It sits at about {own} level in the syllabus, a sensible next step from the topics {sub} did last.")
    return " ".join(sentences[:2])


def area_strengths(
    evidence: dict[int, float],
    concept_area: np.ndarray,
    areas: Sequence[dict],
    prior: np.ndarray,
) -> list[dict]:
    """Estimated understanding per area: the mean over topics with evidence, or the prior where there is none."""
    out = []
    for area in areas:
        members = np.flatnonzero(concept_area == area["id"])
        seen = [evidence[c] for c in members.tolist() if c in evidence]
        value = float(np.mean(seen)) if seen else float(np.mean(prior[members])) if len(members) else 0.0
        out.append({"id": area["id"], "name": area["name"], "value": round(value, 3), "known": bool(seen), "topics": len(seen)})
    return out
