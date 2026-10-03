"""The demo app's model layer: answers questions about one held-out example from an app bundle.

An example is one moment in a test student's history (a cut point). For it the
demo shows the candidate set, the pointer network's path with its attention at
each step, the baselines' paths, what the student actually did next, a DKT
simulation of following each path, and the prerequisite neighbourhood. Every
number comes from the same code the paper's results use.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import torch

from pathtrace import bkt, metrics, prereq
from pathtrace.baselines import MarkovBaseline
from pathtrace.batching import make_batch
from pathtrace.dkt import load_dkt
from pathtrace.preprocess import first_attempts, load_processed
from pathtrace.train import load_checkpoint
from pathtrace.utils import load_json

ATTEMPTS = 3  # simulated attempts per path concept, as in the DKT learning-gain metric
ROLLOUTS = 32
METHODS = ("pointer", "markov", "popularity", "bkt_gain")
METHOD_LABELS = {
    "pointer": "Pointer network",
    "markov": "Markov",
    "popularity": "Popularity",
    "bkt_gain": "Highest BKT gain",
    "actual": "What the student did",
}


def _date(seconds: float) -> str:
    return dt.datetime.fromtimestamp(float(seconds), dt.timezone.utc).date().isoformat()


class DemoError(ValueError):
    """A request the bundle can't answer (unknown student or example, invalid path)."""


class Demo:
    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        root = Path(bundle_dir)
        self.device = torch.device(device)
        self.meta = load_json(root / "meta.json")
        self.concepts = {c["id"]: c for c in load_json(root / "concepts.json")}
        self.proc = load_processed(root / "data")
        self.split = self.meta["source"]["split"]
        self.ex = self.proc.examples[self.split]
        self.inter = self.proc.inter
        self.k = self.ex.targets.shape[1]

        self.model, ckpt = load_checkpoint(root / "model.pt", self.device)
        self.max_history = ckpt["train_config"]["max_history"]
        self.dkt = load_dkt(root / "dkt.pt", self.device)
        self.params = bkt.BKTParams.load(root / "bkt_params.npz")
        self.graph = prereq.PrereqGraph.load(root / "prereq_graph.npz")
        self.parents = [p.tolist() for p in self.graph.parents()]
        self.children: list[list[int]] = [[] for _ in range(self.inter.num_concepts + 1)]
        for s, d in zip(self.graph.src.tolist(), self.graph.dst.tolist()):
            self.children[s].append(d)

        markov = MarkovBaseline()
        with np.load(root / "markov.npz") as z:
            markov.set_counts(z["src"], z["dst"], z["counts"], self.proc.popularity)
        self.scores = {
            "markov": markov.score(self.proc, self.split),
            "popularity": self.proc.popularity[self.ex.candidates].astype(float),
        }
        self.mastery = bkt.candidate_mastery(self.inter, self.ex, self.params)
        self.scores["bkt_gain"] = bkt.expected_gain(self.mastery, self.ex.candidates, self.params)
        self.always, self.need = prereq.example_constraints(self.inter, self.ex, self.graph)
        self.by_student = [np.flatnonzero(self.ex.student == s) for s in range(self.inter.num_students)]
        self.by_student = [idx[np.argsort(self.ex.cut[idx], kind="stable")] for idx in self.by_student]

    # ------------------------------------------------------------------ helpers

    def _concept(self, cid: int) -> dict:
        c = self.concepts[int(cid)]
        return {"id": c["id"], "name": c["name"], "stage": c.get("stage", ""), "level": c.get("level", "")}

    def _check_student(self, student: int) -> None:
        if not 0 <= student < self.inter.num_students:
            raise DemoError(f"unknown student {student}")

    def _check_example(self, example: int) -> None:
        if not 0 <= example < len(self.ex):
            raise DemoError(f"unknown example {example}")

    def _mastery_at(self, student: int, cut: int) -> tuple[dict[int, float], dict[int, list[int]]]:
        """BKT mastery of every concept practised before ``cut``, and its (attempts, correct)."""
        concepts, correct = self.inter.sequence(student)
        mastery: dict[int, float] = {}
        counts: dict[int, list[int]] = {}
        p = self.params
        for c, y in zip(concepts[:cut].tolist(), correct[:cut].tolist()):
            m = mastery.get(c, float(p.p_init[c]))
            mastery[c] = bkt.update(m, y > 0, float(p.p_learn[c]), float(p.p_guess[c]), float(p.p_slip[c]))
            counts.setdefault(c, [0, 0])
            counts[c][0] += 1
            counts[c][1] += int(y > 0)
        return mastery, counts

    # ------------------------------------------------------------------ queries

    def info(self) -> dict:
        return {**self.meta, "method_labels": METHOD_LABELS}

    def students(self) -> list[dict]:
        out = []
        for s in range(self.inter.num_students):
            concepts, correct = self.inter.sequence(s)
            times = self.inter.times(s)
            stages = [self.concepts[int(c)].get("stage", "") for c in np.unique(concepts)]
            out.append(
                {
                    "id": s,
                    "label": str(self.inter.student_uuid[s]),
                    "attempts": int(len(concepts)),
                    "concepts": int(len(np.unique(concepts))),
                    "accuracy": round(float(correct.mean()), 3) if len(correct) else 0.0,
                    "examples": int(len(self.by_student[s])),
                    "first_date": _date(times[0]),
                    "last_date": _date(times[-1]),
                    "stage": max(set(stages), key=stages.count) if stages else "",
                }
            )
        return out

    def student(self, student: int) -> dict:
        self._check_student(student)
        concepts, correct = self.inter.sequence(student)
        times = self.inter.times(student)
        examples = [
            {
                "id": int(e),
                "cut": int(self.ex.cut[e]),
                "date": _date(times[self.ex.cut[e]]),
            }
            for e in self.by_student[student]
        ]
        return {
            "id": student,
            "label": str(self.inter.student_uuid[student]),
            "attempts": [
                {"concept": int(c), "correct": bool(y), "date": _date(t)}
                for c, y, t in zip(concepts.tolist(), correct.tolist(), times.tolist())
            ],
            "examples": examples,
        }

    def example(self, example: int) -> dict:
        self._check_example(example)
        e = example
        student, cut = int(self.ex.student[e]), int(self.ex.cut[e])
        cands = self.ex.candidates[e]
        concepts, correct = self.inter.sequence(student)
        times = self.inter.times(student)

        batch = make_batch(self.inter, self.ex, np.array([e]), self.max_history).to(self.device)
        with torch.no_grad():
            out = self.model.recommend(batch, self.k, mode="greedy")
        pointer = out.actions[0].cpu().numpy()
        step_probs = out.probs[0].cpu().numpy()  # (K, N)

        rng = np.random.default_rng(e)
        paths = {"pointer": pointer}
        for name in ("markov", "popularity", "bkt_gain"):
            paths[name] = metrics.top_k(self.scores[name][e : e + 1], self.k, rng)[0]
        paths["actual"] = self.ex.target_slots[e]
        targets = self.ex.target_slots[e]
        first_group = int(self.ex.first_group[e])

        mastery_now, counts = self._mastery_at(student, cut)
        distinct, firsts = first_attempts(concepts)
        seen = set(distinct[: np.searchsorted(firsts, cut)].tolist())
        candidates = []
        for i, c in enumerate(cands.tolist()):
            unmet = [p for p in self.parents[c] if p not in seen]
            candidates.append(
                {
                    "slot": i,
                    **self._concept(c),
                    "mastery": round(float(self.mastery[e, i]), 3),
                    "bkt_gain": round(float(self.scores["bkt_gain"][e, i]), 4),
                    "popularity": int(self.proc.popularity[c]),
                    "markov": round(float(self.scores["markov"][e, i]), 4),
                    "unmet_prereqs": [self._concept(p) for p in unmet],
                    "target_rank": int(np.flatnonzero(targets == i)[0]) + 1 if i in targets else None,
                }
            )

        def describe(name: str, slots: np.ndarray) -> dict:
            slots = np.asarray(slots)
            return {
                "name": name,
                "label": METHOD_LABELS[name],
                "slots": slots.tolist(),
                "concepts": [self._concept(cands[s]) for s in slots],
                "hits": int(np.isin(slots, targets).sum()),
                "first_step": bool(slots[0] in targets[:first_group]),
                "violations": int(prereq.path_violations(slots[None], self.always[e : e + 1], self.need[e : e + 1])[0]),
                "bkt_reward": round(float(self.scores["bkt_gain"][e, slots].sum()), 4),
            }

        recent_ids = list(dict.fromkeys(reversed(concepts[:cut].tolist())))[:12]
        return {
            "id": e,
            "student": student,
            "cut": cut,
            "date": _date(times[cut]),
            "history": {
                "attempts": cut,
                "concepts": len(seen),
                "accuracy": round(float(correct[:cut].mean()), 3) if cut else 0.0,
                "recent": [
                    {
                        **self._concept(c),
                        "mastery": round(mastery_now[c], 3),
                        "attempts": counts[c][0],
                        "correct": counts[c][1],
                    }
                    for c in recent_ids
                ],
            },
            "candidates": candidates,
            "paths": {name: describe(name, slots) for name, slots in paths.items()},
            "pointer_steps": [[round(float(p), 4) for p in row] for row in step_probs],
            "first_group": first_group,
        }

    @torch.no_grad()
    def simulate(self, example: int, paths: dict[str, list[int]], rollouts: int = ROLLOUTS, seed: int = 0) -> dict:
        """DKT rollout of each path: mean predicted success on the candidates after every attempt.

        As in the paper's learning-gain metric, each path concept gets ``ATTEMPTS``
        attempts whose answers are sampled from DKT, averaged over ``rollouts``, and
        every path sees the same random draws.
        """
        self._check_example(example)
        n = self.ex.candidates.shape[1]
        for name, slots in paths.items():
            if not slots or len(slots) > self.k or len(set(slots)) != len(slots) or not all(0 <= s < n for s in slots):
                raise DemoError(f"path {name!r} must be 1-{self.k} distinct slots in 0..{n - 1}")
        batch = make_batch(self.inter, self.ex, np.array([example]), 200).to(self.device)
        encoded = self.dkt.encode(batch.hist_concepts, batch.hist_correct, batch.lengths)
        cands = batch.candidates[0]
        before_all = self.dkt.probs(encoded)[0]
        before = float(before_all[cands].mean())
        generator = torch.Generator(device=self.device)
        result = {}
        for name, slots in paths.items():
            generator.manual_seed(seed + example)
            state = tuple(s.repeat_interleave(rollouts, dim=1) for s in encoded)
            curve = [before]
            per_concept = []
            for slot in slots:
                concept = cands[slot].repeat(rollouts)
                start = float(self.dkt.probs(state)[:, concept[0]].mean())
                for _ in range(ATTEMPTS):
                    p = self.dkt.probs(state).gather(1, concept.unsqueeze(1)).squeeze(1)
                    answers = (torch.rand(p.shape, generator=generator, device=self.device) < p).float()
                    state = self.dkt.step(concept, answers, state)
                    curve.append(float(self.dkt.probs(state)[:, cands].mean()))
                per_concept.append(
                    {"slot": slot, **self._concept(int(cands[slot])), "before": round(start, 4),
                     "after": round(float(self.dkt.probs(state)[:, concept[0]].mean()), 4)}
                )
            after = curve[-1]
            result[name] = {
                "before": round(before, 4),
                "after": round(after, 4),
                "gain": round((after - before) / max(1.0 - before, 1e-6), 4),
                "curve": [round(v, 4) for v in curve],
                "concepts": per_concept,
            }
        return {"attempts_per_concept": ATTEMPTS, "rollouts": rollouts, "paths": result}

    def neighbourhood(self, example: int, max_nodes: int = 60) -> dict:
        """Candidates, their prerequisites and the edges between them, with mastery and status."""
        self._check_example(example)
        student, cut = int(self.ex.student[example]), int(self.ex.cut[example])
        cands = self.ex.candidates[example].tolist()
        mastery_now, _ = self._mastery_at(student, cut)
        nodes = list(dict.fromkeys(cands + [p for c in cands for p in self.parents[c]]))[:max_nodes]
        keep = set(nodes)
        edges = [
            {"source": p, "target": c}
            for c in nodes
            for p in self.parents[c]
            if p in keep
        ]
        return {
            "nodes": [
                {
                    **self._concept(c),
                    "candidate": c in cands,
                    "seen": c in mastery_now,
                    "mastery": round(mastery_now.get(c, float(self.params.p_init[c])), 3),
                }
                for c in nodes
            ],
            "edges": edges,
        }

    def tutor_context(self, example: int) -> str:
        """Plain-text facts about the example for the tutor prompt (no student identifiers)."""
        ex = self.example(example)
        lines = [
            f"Student history before this point: {ex['history']['attempts']} attempts on "
            f"{ex['history']['concepts']} concepts, {ex['history']['accuracy']:.0%} correct.",
            "Recently practised (BKT mastery): "
            + "; ".join(f"{r['name']} ({r['mastery']:.0%})" for r in ex["history"]["recent"][:6]),
            f"The pointer network chose this path from {len(ex['candidates'])} candidate concepts, with the "
            "probability it gave each pick at its step:",
        ]
        for step, slot in enumerate(ex["paths"]["pointer"]["slots"]):
            cand = ex["candidates"][slot]
            prereqs = ", ".join(p["name"] for p in cand["unmet_prereqs"]) or "none"
            lines.append(
                f"  {step + 1}. {cand['name']}: probability {ex['pointer_steps'][step][slot]:.0%}, "
                f"BKT mastery {cand['mastery']:.0%}, expected BKT gain {cand['bkt_gain']:.3f}, "
                f"unmet inferred prerequisites: {prereqs}"
            )
        for name in ("markov", "popularity"):
            path = ex["paths"][name]
            lines.append(f"{path['label']} baseline would pick: " + ", ".join(c["name"] for c in path["concepts"]))
        return "\n".join(lines)
