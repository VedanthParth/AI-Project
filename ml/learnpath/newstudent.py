"""The demo's "New student" page: questionnaire answers in, every step of the recommendation out.

A plan is computed from scratch on each request (the client sends the answers
plus any practice so far), so changing an answer or recording practice shows
how the recommendation changes:

1. answers -> history: ``onboarding.checklist_history`` (units marked good or
   bad, the exercises named as finished last, the earlier exercises of their
   unit taken as done), then each practice attempt in order;
2. history -> BKT mastery of every concept, replayed attempt by attempt;
3. eligible concepts: the stage's concepts not in the history;
4. shortlist: the Markov scores of every eligible concept (the same scorer as
   ``learnpath catalogue``), top N;
5. the pointer network orders the shortlist step by step (its attention over
   the shortlist at each step), and so does the RL-fine-tuned policy;
6. each path's expected BKT gain, inferred prerequisites it skips, and a DKT
   simulation of practising it;
7. a review list, by rule rather than model: concepts already worked on whose
   BKT mastery is below ``REVIEW_BELOW`` (the models only recommend new ones).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from learnpath import bkt
from learnpath.batching import Batch
from learnpath.demo import ATTEMPTS, ROLLOUTS, Demo, DemoError
from learnpath.onboarding import BAD, GOOD, Topics, checklist_history
from learnpath.preprocess import MarkovScorer
from learnpath.train import load_checkpoint
from learnpath.utils import load_json

STAGES = ("elementary", "junior")
RATINGS = ("strong", "okay", "weak")
MAX_CURRENT = 3
REVIEW_BELOW = 0.6  # BKT mastery under which a concept already worked on is listed for review


class NewStudentPlanner:
    def __init__(self, demo: Demo, bundle_dir: str | Path):
        root = Path(bundle_dir)
        if not (root / "topics.json").exists():
            raise DemoError("this bundle has no topics.json; export it with --content to enable the New student page")
        self.demo = demo
        self.topics_json = load_json(root / "topics.json")
        n = demo.inter.num_concepts + 1
        self.position = np.array([demo.concepts.get(c, {}).get("position", 0.5) for c in range(n)])
        self.stage = np.array([demo.concepts.get(c, {}).get("stage", "") for c in range(n)])
        units = self.topics_json["units"]
        concept_unit = np.full(n, -1, dtype=np.int64)
        for u in units:
            concept_unit[u["concepts"]] = u["id"]
        import pandas as pd

        self.topics = Topics(
            strands=pd.DataFrame(self.topics_json["strands"]),
            units=pd.DataFrame({"id": [u["id"] for u in units], "concepts": [np.asarray(u["concepts"], dtype=np.int64) for u in units]}),
            concept_unit=concept_unit,
        )
        with np.load(root / "markov.npz") as z:
            self.markov = MarkovScorer(z["src"], z["dst"], z["counts"], demo.proc.popularity)
        self.models = {"pointer": (demo.model, demo.max_history, demo.meta["source"]["checkpoint"])}
        self.shortlist_size = demo.ex.candidates.shape[1]
        if (root / "shortlist_model.pt").exists():
            model, ckpt = load_checkpoint(root / "shortlist_model.pt", demo.device)
            self.models["pointer"] = (model, ckpt["train_config"]["max_history"], demo.meta.get("shortlist_model") or "shortlist model")
            self.shortlist_size = int(ckpt["data_config"]["num_candidates"])
        if (root / "rl.pt").exists():
            model, ckpt = load_checkpoint(root / "rl.pt", demo.device)
            self.models["rl"] = (model, ckpt["train_config"]["max_history"], (demo.meta.get("rl") or {}).get("checkpoint", "rl"))

    # ------------------------------------------------------------------ questionnaire

    def questionnaire(self) -> dict:
        """Strands and units per stage, each unit with its exercises in curriculum order."""
        units = []
        for u in self.topics_json["units"]:
            if u["stage"] not in STAGES:
                continue
            order = sorted(u["concepts"], key=lambda c: self.position[c])
            units.append(
                {
                    "id": u["id"], "strand": u["strand"], "name": u["name"], "stage": u["stage"],
                    "exercises": [{"id": c, "name": self.demo.concepts[c]["name"], "level": self.demo.concepts[c].get("level", "")} for c in order],
                }
            )
        strands = [s for s in self.topics_json["strands"] if s["stage"] in STAGES]
        return {"stages": list(STAGES), "ratings": list(RATINGS), "strands": strands, "units": units, "max_current": MAX_CURRENT}

    # ------------------------------------------------------------------ plan

    def _marks(self, stage: str, strands: dict[int, str], units: dict[int, str]) -> dict[int, str]:
        unit_rows = {u["id"]: u for u in self.topics_json["units"]}
        marks = {}
        for strand, rating in strands.items():
            if rating not in RATINGS:
                raise DemoError(f"strand ratings must be one of {RATINGS}, got {rating!r}")
            if rating == "strong":
                marks.update({u["id"]: GOOD for u in unit_rows.values() if u["strand"] == strand and u["stage"] == stage})
        for unit, mark in units.items():
            if unit not in unit_rows or mark not in (GOOD, BAD):
                raise DemoError(f"unknown unit {unit} or mark {mark!r}")
            marks[unit] = mark
        return marks

    def _sources(self, marks: dict[int, str], current: list[int]) -> dict[int, str]:
        """Why each history concept is there, for the page."""
        names = {u["id"]: u["name"] for u in self.topics_json["units"]}
        source = {}
        for unit, mark in marks.items():
            for c in self.topics.representatives(unit, self.demo.proc.popularity, 4).tolist():
                source[c] = f"you marked {names[unit]} as {'confident' if mark == GOOD else 'needs help'}"
        if current:
            unit = int(self.topics.concept_unit[current[-1]])
            if unit >= 0:
                for c in self.topics.units.concepts[unit].tolist():
                    if self.position[c] < self.position[current[-1]] and c not in current:
                        source[c] = f"assumed done: comes before your last exercise in {names[unit]}"
        for c in current:
            source[c] = "you finished this last"
        return source

    @torch.no_grad()
    def plan(self, stage: str, strands: dict[int, str], units: dict[int, str], current: list[int], practice: list[tuple[int, bool]]) -> dict:
        demo = self.demo
        if stage not in STAGES:
            raise DemoError(f"stage must be one of {STAGES}")
        if len(current) > MAX_CURRENT or any(c not in demo.concepts for c in current):
            raise DemoError(f"name at most {MAX_CURRENT} known exercises as finished last")
        if any(c not in demo.concepts for c, _ in practice):
            raise DemoError("practice lists an unknown exercise")
        marks = self._marks(stage, strands, units)
        concepts, correct = checklist_history(marks, self.topics, demo.proc.popularity, self.position, current=current)
        if practice:
            concepts = np.concatenate([concepts, [c for c, _ in practice]]).astype(np.int32)
            correct = np.concatenate([correct, [int(y) for _, y in practice]]).astype(np.int8)
        if len(concepts) == 0:
            raise DemoError(
                "rate an area of maths as strong, mark a chapter, or name an exercise you finished, so there is something to go on"
            )
        source = self._sources(marks, current)
        for c, _ in practice:
            source[c] = "practised here"

        mastery = self._mastery(concepts, correct)
        eligible = (self.stage == stage) & ~np.isin(np.arange(len(self.stage)), concepts)
        eligible[0] = False
        scores = self.markov(_started(concepts))
        scores[~eligible] = -np.inf
        n = min(self.shortlist_size, int(eligible.sum()))
        shortlist = np.argsort(-scores, kind="stable")[:n]
        gain = (1 - mastery[shortlist]) * self.demo.params.p_learn[shortlist]
        seen = set(concepts.tolist())

        paths = {"markov": {"slots": list(range(demo.k)), "steps": None, "model": "first-attempt transitions"}}
        for name, (model, max_history, label) in self.models.items():
            out = model.recommend(_batch(concepts, correct, shortlist, demo.k, max_history, demo.device), demo.k)
            paths[name] = {
                "slots": out.actions[0].tolist(),
                "steps": [[round(float(p), 4) for p in row] for row in out.probs[0].cpu().numpy()],
                "model": label,
            }
        described = {}
        for name, p in paths.items():
            slots = p["slots"]
            items = []
            for step, slot in enumerate(slots):
                c = int(shortlist[slot])
                earlier = {int(shortlist[s]) for s in slots[:step]}
                unmet = [q for q in demo.parents[c] if q not in seen and q not in earlier]
                items.append({**self._concept(c), "slot": slot, "unmet_prereqs": [self._concept(q) for q in unmet]})
            described[name] = {
                **p,
                "label": {"pointer": "Pointer network", "rl": "RL-tuned policy", "markov": "Markov"}[name],
                "concepts": items,
                "bkt_gain": round(float(gain[slots].sum()), 4),
                "violations": sum(1 for it in items if it["unmet_prereqs"]),
            }

        unit_summary = {}
        for c in seen:
            u = int(self.topics.concept_unit[c])
            if u >= 0:
                unit_summary.setdefault(u, []).append(mastery[c])
        unit_names = {u["id"]: u for u in self.topics_json["units"]}
        return {
            "pipeline": {
                "catalogue": int(demo.inter.num_concepts),
                "stage_concepts": int((self.stage == stage).sum()),
                "history_attempts": int(len(concepts)),
                "history_concepts": int(len(seen)),
                "eligible": int(eligible.sum()),
                "shortlist_size": int(n),
                "models": {name: label for name, (_, _, label) in self.models.items()},
            },
            "history": [
                {**self._concept(c), "source": source.get(c, ""), "mastery": round(float(mastery[c]), 3),
                 "attempts": int((concepts == c).sum()), "correct": int(correct[concepts == c].sum())}
                for c in _started(concepts).tolist()
            ],
            "units": [
                {"id": u, "name": unit_names[u]["name"], "mark": marks.get(u, "inferred"), "mastery": round(float(np.mean(v)), 3)}
                for u, v in sorted(unit_summary.items(), key=lambda kv: np.mean(kv[1]))
            ],
            "shortlist": [
                {**self._concept(int(c)), "rank": i + 1, "markov": round(float(scores[c]), 5),
                 "mastery": round(float(mastery[c]), 3), "bkt_gain": round(float(gain[i]), 4),
                 "weak_unit": marks.get(int(self.topics.concept_unit[c])) == BAD}
                for i, c in enumerate(shortlist.tolist())
            ],
            "paths": described,
            # A rule, not a model: the models only recommend new concepts, so concepts already worked
            # on whose BKT mastery is still low are listed for review separately.
            "review": [
                {**self._concept(c), "mastery": round(float(mastery[c]), 3), "source": source.get(c, "")}
                for c in sorted(seen, key=lambda c: mastery[c])
                if mastery[c] < REVIEW_BELOW
            ][:5],
            "review_below": REVIEW_BELOW,
            "simulation": self._simulate(concepts, correct, shortlist, {k: v["slots"] for k, v in described.items()}),
        }

    # ------------------------------------------------------------------ helpers

    def _concept(self, c: int) -> dict:
        info = self.demo._concept(c)
        u = int(self.topics.concept_unit[c])
        unit = self.topics_json["units"][u]["name"] if u >= 0 else ""
        return {**info, "unit": unit}

    def _mastery(self, concepts: np.ndarray, correct: np.ndarray) -> np.ndarray:
        p = self.demo.params
        mastery = p.p_init.astype(np.float64).copy()
        for c, y in zip(concepts.tolist(), correct.tolist()):
            mastery[c] = bkt.update(mastery[c], y > 0, float(p.p_learn[c]), float(p.p_guess[c]), float(p.p_slip[c]))
        return mastery

    @torch.no_grad()
    def _simulate(self, concepts, correct, shortlist, paths: dict[str, list[int]], rollouts: int = ROLLOUTS) -> dict:
        """DKT rollout of each path, as on the Real students page: mean predicted success on the shortlist."""
        dkt, device = self.demo.dkt, self.demo.device
        hist = torch.as_tensor(concepts[-200:], dtype=torch.long, device=device).unsqueeze(0)
        right = torch.as_tensor(correct[-200:], dtype=torch.long, device=device).unsqueeze(0)
        encoded = dkt.encode(hist, right, torch.tensor([hist.size(1)]))
        cands = torch.as_tensor(shortlist, dtype=torch.long, device=device)
        before = float(dkt.probs(encoded)[0][cands].mean())
        generator = torch.Generator(device=device)
        result = {}
        for name, slots in paths.items():
            generator.manual_seed(0)
            state = tuple(s.repeat_interleave(rollouts, dim=1) for s in encoded)
            curve = [before]
            for slot in slots:
                concept = cands[slot].repeat(rollouts)
                for _ in range(ATTEMPTS):
                    p = dkt.probs(state).gather(1, concept.unsqueeze(1)).squeeze(1)
                    answers = (torch.rand(p.shape, generator=generator, device=device) < p).float()
                    state = dkt.step(concept, answers, state)
                    curve.append(float(dkt.probs(state)[:, cands].mean()))
            result[name] = {
                "before": round(before, 4), "after": round(curve[-1], 4),
                "gain": round((curve[-1] - before) / max(1.0 - before, 1e-6), 4), "curve": [round(v, 4) for v in curve],
            }
        return {"attempts_per_concept": ATTEMPTS, "rollouts": rollouts, "paths": result}


def _started(concepts: np.ndarray) -> np.ndarray:
    """Distinct concepts in first-attempt order."""
    _, first = np.unique(concepts, return_index=True)
    return concepts[np.sort(first)]


def _batch(concepts: np.ndarray, correct: np.ndarray, candidates: np.ndarray, k: int, max_history: int, device) -> Batch:
    hist, right = concepts[-max_history:], correct[-max_history:]
    return Batch(
        hist_concepts=torch.as_tensor(hist, dtype=torch.long).unsqueeze(0),
        hist_correct=torch.as_tensor(right, dtype=torch.float32).unsqueeze(0),
        lengths=torch.tensor([len(hist)]),
        candidates=torch.as_tensor(candidates, dtype=torch.long).unsqueeze(0),
        target_slots=torch.zeros((1, k), dtype=torch.long),
        targets=torch.zeros((1, k), dtype=torch.long),
        first_group=torch.ones(1, dtype=torch.long),
        index=np.array([0]),
    ).to(device)
