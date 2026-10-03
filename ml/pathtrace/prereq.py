"""Prerequisite graph inferred from the order in which students start concepts.

For every pair of concepts (a, b) that training students both attempted:

- support    = students who started a and b in different time windows
- precedence = share of those students who started a first
- proximity  = share of those students who started b within the next
               ``proximity_window`` new concepts after a

An edge a -> b is kept when support, precedence and proximity all clear their
thresholds and a's average (normalised) first-attempt position is earlier
than b's. That last rule makes the graph acyclic by construction. A
transitive reduction then drops edges implied by longer paths.

Uses:

- ``example_constraints`` turns the graph into per-example tables saying
  which candidates have unmet prerequisites (given the history) and which
  other candidates would satisfy them earlier in a path;
- ``path_violations`` counts, for any ordered path, the concepts started
  before all their prerequisites (V(P) in the reward).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pathtrace.preprocess import Examples, Interactions, Processed, first_attempts
from pathtrace.utils import save_json


@dataclass
class PrereqGraph:
    src: np.ndarray  # (E,) prerequisite concept ids
    dst: np.ndarray  # (E,) dependent concept ids
    support: np.ndarray  # (E,)
    precedence: np.ndarray  # (E,)
    proximity: np.ndarray  # (E,)
    num_concepts: int

    def parents(self) -> list[np.ndarray]:
        order = np.argsort(self.dst, kind="stable")
        bounds = np.searchsorted(self.dst[order], np.arange(self.num_concepts + 2))
        src = self.src[order]
        return [src[bounds[c] : bounds[c + 1]] for c in range(self.num_concepts + 1)]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            src=self.src,
            dst=self.dst,
            support=self.support,
            precedence=self.precedence,
            proximity=self.proximity,
            num_concepts=self.num_concepts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrereqGraph":
        with np.load(path) as z:
            return cls(
                z["src"], z["dst"], z["support"], z["precedence"], z["proximity"], int(z["num_concepts"])
            )


# --------------------------------------------------------------------------- inference


def _pair_counts(
    inter: Interactions, students: np.ndarray, window: int, chunk: int = 2000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense (C+1)^2 counts: a started strictly before b, and b within ``window`` new concepts of a.

    Also the mean normalised first-attempt position of each concept.
    """
    size = inter.num_concepts + 1
    before = np.zeros(size * size, dtype=np.int64)
    near = np.zeros(size * size, dtype=np.int64)
    pos_sum = np.zeros(size)
    pos_count = np.zeros(size)
    before_idx: list[np.ndarray] = []
    near_idx: list[np.ndarray] = []

    def flush() -> None:
        nonlocal before, near
        if before_idx:
            before += np.bincount(np.concatenate(before_idx), minlength=size * size)
            near += np.bincount(np.concatenate(near_idx), minlength=size * size)
            before_idx.clear()
            near_idx.clear()

    for n, s in enumerate(students, start=1):
        distinct, firsts = first_attempts(inter.sequence(s)[0])
        m = len(distinct)
        if m < 2:
            continue
        windows = inter.times(s)[firsts]
        pos_sum[distinct] += np.arange(m) / (m - 1)
        pos_count[distinct] += 1
        i, j = np.triu_indices(m, 1)
        strict = windows[i] < windows[j]
        i, j = i[strict], j[strict]
        pairs = distinct[i].astype(np.int64) * size + distinct[j]
        before_idx.append(pairs)
        near_idx.append(pairs[(j - i) <= window])
        if n % chunk == 0:
            flush()
    flush()
    with np.errstate(invalid="ignore"):
        mean_pos = np.where(pos_count > 0, pos_sum / pos_count, np.nan)
    return before.reshape(size, size), near.reshape(size, size), mean_pos, pos_count


def _transitive_reduction(src: np.ndarray, dst: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Mask of edges not implied by a longer path. Edges must go forward in ``rank``."""
    children: dict[int, list[int]] = {}
    for a, b in zip(src.tolist(), dst.tolist()):
        children.setdefault(a, []).append(b)
    reach: dict[int, int] = {}  # node -> bitmask of nodes reachable from it
    for node in sorted(children, key=lambda c: -rank[c]):
        mask = 0
        for child in children[node]:
            mask |= (1 << child) | reach.get(child, 0)
        reach[node] = mask
    keep = np.ones(len(src), dtype=bool)
    for e, (a, b) in enumerate(zip(src.tolist(), dst.tolist())):
        via_others = 0
        for child in children[a]:
            if child != b:
                via_others |= reach.get(child, 0)
        keep[e] = not (via_others >> b) & 1
    return keep


def infer_graph(
    inter: Interactions,
    students: np.ndarray,
    *,
    min_support: int = 30,
    min_precedence: float = 0.9,
    min_proximity: float = 0.2,
    proximity_window: int = 5,
    reduce: bool = True,
) -> tuple[PrereqGraph, dict]:
    before, near, mean_pos, _ = _pair_counts(inter, students, proximity_window)
    support = before + before.T
    with np.errstate(invalid="ignore", divide="ignore"):
        precedence = before / support
        proximity = near / support
    forward = mean_pos[:, None] < mean_pos[None, :]
    candidate = (support >= min_support) & (precedence >= min_precedence) & (proximity >= min_proximity) & forward
    candidate[0, :] = candidate[:, 0] = False
    src, dst = np.nonzero(candidate)
    info = {"edges_before_reduction": int(len(src))}
    if reduce and len(src):
        rank = np.nan_to_num(mean_pos, nan=np.inf)
        keep = _transitive_reduction(src, dst, rank)
        src, dst = src[keep], dst[keep]
    graph = PrereqGraph(
        src=src.astype(np.int32),
        dst=dst.astype(np.int32),
        support=support[src, dst],
        precedence=precedence[src, dst],
        proximity=proximity[src, dst],
        num_concepts=inter.num_concepts,
    )
    in_graph = np.zeros(inter.num_concepts + 1, dtype=bool)
    in_graph[src] = in_graph[dst] = True
    in_degree = np.bincount(dst, minlength=inter.num_concepts + 1)[1:]
    info.update(
        {
            "edges": int(len(src)),
            "concepts_in_graph": int(in_graph.sum()),
            "concepts_with_prerequisites": int((in_degree > 0).sum()),
            "mean_prerequisites_when_any": float(in_degree[in_degree > 0].mean()) if (in_degree > 0).any() else 0.0,
        }
    )
    return graph, info


# --------------------------------------------------------------------------- validation


def compare_with_content(graph: PrereqGraph, concept_names: np.ndarray, content: pd.DataFrame, seed: int = 0) -> dict:
    """How well edges agree with Junyi's topic hierarchy and learning stages, vs random pairs."""
    info = content.set_index("ucid").reindex(concept_names[1:])
    stage_rank = {"elementary": 0, "junior": 1, "senior": 2}
    columns = {
        "same_level2_topic": info["level2_id"].to_numpy(),
        "same_level3_topic": info["level3_id"].to_numpy(),
        "same_level4_topic": info["level4_id"].to_numpy(),
    }
    stage = info["learning_stage"].map(stage_rank).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rand_src = rng.integers(0, graph.num_concepts, 20_000)
    rand_dst = rng.integers(0, graph.num_concepts, 20_000)
    src, dst = graph.src - 1, graph.dst - 1

    def stage_backwards(a: np.ndarray, b: np.ndarray) -> float:
        ok = ~np.isnan(stage[a]) & ~np.isnan(stage[b])
        return float((stage[a][ok] > stage[b][ok]).mean()) if ok.any() else float("nan")

    out = {}
    for name, values in columns.items():
        out[name] = {
            "edges": float((values[src] == values[dst]).mean()) if len(src) else float("nan"),
            "random_pairs": float((values[rand_src] == values[rand_dst]).mean()),
        }
    out["later_stage_before_earlier"] = {
        "edges": stage_backwards(src, dst) if len(src) else float("nan"),
        "random_pairs": stage_backwards(rand_src, rand_dst),
    }
    return out


# --------------------------------------------------------------------------- constraints


def example_constraints(
    inter: Interactions, ex: Examples, graph: PrereqGraph
) -> tuple[np.ndarray, np.ndarray]:
    """Per-example prerequisite tables over the candidate slots.

    ``always[e, i]``: candidate i has an unmet prerequisite that is neither
    in the history nor among the candidates, so starting it always violates.
    ``need[e, i, j]``: candidate j is an unmet prerequisite of candidate i,
    so i violates unless j comes earlier in the path.
    """
    parents = [p.tolist() for p in graph.parents()]
    n = ex.candidates.shape[1]
    always = np.zeros((len(ex), n), dtype=bool)
    need = np.zeros((len(ex), n, n), dtype=bool)
    current, distinct, firsts = -1, None, None
    for e in range(len(ex)):
        if ex.student[e] != current:
            current = int(ex.student[e])
            distinct, firsts = first_attempts(inter.sequence(current)[0])
        seen = set(distinct[: np.searchsorted(firsts, ex.cut[e])].tolist())
        cands = ex.candidates[e].tolist()
        slot = {c: i for i, c in enumerate(cands)}
        for i, c in enumerate(cands):
            for p in parents[c]:
                if p in seen:
                    continue
                if p in slot:
                    need[e, i, slot[p]] = True
                else:
                    always[e, i] = True
    return always, need


def path_violations(paths: np.ndarray, always: np.ndarray, need: np.ndarray) -> np.ndarray:
    """Number of concepts in each path started before all their prerequisites, shape (E,)."""
    rows = np.arange(len(paths))
    picked = np.zeros(need.shape[:2], dtype=bool)
    count = np.zeros(len(paths), dtype=np.int64)
    for t in range(paths.shape[1]):
        slot = paths[:, t]
        unmet = need[rows, slot] & ~picked
        count += always[rows, slot] | unmet.any(axis=1)
        picked[rows, slot] = True
    return count


# --------------------------------------------------------------------------- command


def infer_and_report(
    proc: Processed,
    out_dir: str | Path,
    content_path: str | Path | None = None,
    log: Callable[[str], None] = print,
    **kwargs,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph, info = infer_graph(proc.inter, proc.splits["train"], **kwargs)
    graph.save(out / "prereq_graph.npz")
    log(f"prerequisite graph: {info['edges']} edges over {info['concepts_in_graph']} concepts")
    summary: dict = {"graph": info, "settings": kwargs, "violations": {}}

    content = pd.read_csv(content_path) if content_path else None
    if content is not None:
        summary["agreement_with_content"] = compare_with_content(graph, proc.inter.concept_names, content)
        names = content.set_index("ucid")["content_pretty_name"].reindex(proc.inter.concept_names).to_numpy()
        top = np.argsort(-graph.support)[:15]
        summary["top_edges"] = [
            {"from": names[graph.src[e]], "to": names[graph.dst[e]], "support": int(graph.support[e])} for e in top
        ]

    k = proc.data_cfg.path_len
    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        ex = proc.examples[split]
        if len(ex) == 0:
            continue
        always, need = example_constraints(proc.inter, ex, graph)
        random_paths = np.argsort(rng.random(ex.candidates.shape), axis=1)[:, :k]
        summary["violations"][split] = {
            "student_paths": float(path_violations(ex.target_slots, always, need).mean()),
            "random_paths": float(path_violations(random_paths, always, need).mean()),
            "candidates_with_unmet_prerequisites": float((always | need.any(axis=2)).mean()),
        }
    save_json(summary, out / "prereq_summary.json")
    return summary
