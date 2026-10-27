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
5. the recommender (a pointer network after RL fine-tuning) orders the
   shortlist step by step (its attention over the shortlist at each step);
6. each path's expected BKT gain, inferred prerequisites it skips, and a DKT
   simulation of practising it;
7. a review list, by rule rather than model: concepts already worked on whose
   BKT mastery is below ``REVIEW_BELOW`` (the models only recommend new ones).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from learnpath import bkt, explain
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
        answered = len(concepts) - len(practice)  # attempts built from the answers, before any practice
        before = self._mastery(concepts[:answered], correct[:answered]) if practice else mastery
        eligible = (self.stage == stage) & ~np.isin(np.arange(len(self.stage)), concepts)
        eligible[0] = False
        options = np.flatnonzero(eligible)
        scores = self.markov(_started(concepts))
        scores[~eligible] = -np.inf
        n = min(self.shortlist_size, len(options))
        shortlist = np.argsort(-scores, kind="stable")[:n]
        expected_gain = (1 - mastery) * self.demo.params.p_learn
        recent = explain.scaled(np.where(eligible, scores, 0.0), options)
        level = explain.syllabus_level(self.position, _started(concepts))
        gain = explain.ranked(expected_gain, options)
        seen = set(concepts.tolist())

        # Three ways of choosing three topics: the recommender orders the shortlist; the usual next
        # step is the shortlist's top three; the most common topics ignore the student entirely.
        ids = {
            "markov": shortlist[: demo.k].tolist(),
            "popularity": options[np.argsort(-demo.proc.popularity[options], kind="stable")[: demo.k]].tolist(),
        }
        steps = {}
        for key, (model, max_history, _) in self.models.items():
            out = model.recommend(_batch(concepts, correct, shortlist, demo.k, max_history, demo.device), demo.k)
            ids[key] = shortlist[out.actions[0].cpu().numpy()].tolist()
            steps[key] = out.probs[0].cpu().numpy()
        simulation = self._simulate(concepts, correct, shortlist, ids)
        best_progress = max(max(s["gain"], 0.0) for s in simulation["paths"].values()) or 1.0
        best_gain = max(float(expected_gain[path].sum()) for path in ids.values())
        slot_of = {int(c): i for i, c in enumerate(shortlist.tolist())}

        paths = {}
        for key, path in ids.items():
            items, mentioned = [], set()  # name a "needs help" chapter on its first topic only
            for step, c in enumerate(path):
                earlier = path[:step]
                topic = self._concept(c)
                weak = marks.get(int(self.topics.concept_unit[c])) == BAD and topic["unit"] not in mentioned
                mentioned.add(topic["unit"])
                items.append(
                    {
                        **topic,
                        "slot": slot_of.get(c),
                        "mastery": round(float(mastery[c]), 3),
                        "unmet_prereqs": [self._concept(q) for q in explain.unmet(c, demo.parents, seen | set(earlier))],
                        "factors": explain.topic_factors(
                            c, done=seen | set(earlier), parents=demo.parents, mastery=mastery, recent=recent,
                            level=level, gain=gain,
                            sure=float(steps[key][step, slot_of[c]]) if key in steps else None,
                        ),
                        "reason": explain.topic_reason(
                            c, history=seen, earlier=earlier, parents=demo.parents, mastery=mastery, recent=recent,
                            name=lambda q: demo.concepts[q]["name"],
                            needs_help=weak, chapter=topic["unit"],
                        ),
                    }
                )
            paths[key] = {
                "concepts": items,
                "steps": [[round(float(p), 4) for p in row] for row in steps[key]] if key in steps else None,
                "model": self.models[key][2] if key in self.models else key,
                "violations": sum(1 for it in items if it["unmet_prereqs"]),
                "bkt_gain": round(float(expected_gain[path].sum()), 4),
                "profile": explain.path_profile(
                    path, history=seen, parents=demo.parents, mastery=mastery, recent=recent, level=level,
                    expected_gain=expected_gain, best_gain=best_gain,
                    progress=max(simulation["paths"][key]["gain"], 0.0) / best_progress,
                ),
            }

        # Estimated understanding of each topic practised in the app, after every attempt.
        trace: dict[int, list[float]] = {}
        replay = before.copy()
        p = demo.params
        for c, right in practice:
            trace.setdefault(c, [float(replay[c])])
            replay[c] = bkt.update(replay[c], right, float(p.p_learn[c]), float(p.p_guess[c]), float(p.p_slip[c]))
            trace[c].append(float(replay[c]))

        areas = [s for s in self.topics_json["strands"] if s["stage"] == stage]
        answered_concepts = set(concepts[:answered].tolist())
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
                "eligible": int(len(options)),
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
            "areas": {
                "before": explain.area_strengths(
                    {c: float(before[c]) for c in answered_concepts}, demo.concept_area, areas, p.p_init
                ),
                "now": explain.area_strengths({c: float(mastery[c]) for c in seen}, demo.concept_area, areas, p.p_init),
            },
            "practice_trace": [
                {**self._concept(c), "points": [round(v, 3) for v in points]} for c, points in trace.items()
            ],
            "shortlist": [
                {**self._concept(int(c)), "rank": i + 1, "markov": round(float(scores[c]), 5),
                 "mastery": round(float(mastery[c]), 3), "bkt_gain": round(float(expected_gain[c]), 4),
                 "weak_unit": marks.get(int(self.topics.concept_unit[c])) == BAD}
                for i, c in enumerate(shortlist.tolist())
            ],
            "paths": paths,
            # A rule, not a model: the models only recommend new concepts, so concepts already worked
            # on whose BKT mastery is still low are listed for review separately.
            "review": [
                {**self._concept(c), "mastery": round(float(mastery[c]), 3), "source": source.get(c, "")}
                for c in sorted(seen, key=lambda c: mastery[c])
                if mastery[c] < REVIEW_BELOW
            ][:5],
            "review_below": REVIEW_BELOW,
            "simulation": simulation,
        }

    def facts(self, plan: dict) -> str:
        """Plain-text facts about a plan for the tutor prompt (no names or other identifiers)."""
        confident = [u["name"] for u in plan["units"] if u["mark"] == GOOD]
        needs_help = [u["name"] for u in plan["units"] if u["mark"] == BAD]
        last = [h["name"] for h in plan["history"] if h["source"] == "you finished this last"]
        lines = []
        if confident:
            lines.append("Chapters the student is confident in: " + ", ".join(confident) + ".")
        if needs_help:
            lines.append("Chapters the student said need help: " + ", ".join(needs_help) + ".")
        if last:
            lines.append("Exercises the student finished most recently: " + ", ".join(last) + ".")
        lines.append("The recommended next three topics, in order, each with the reason the app shows:")
        for step, item in enumerate(plan["paths"]["pointer"]["concepts"]):
            lines.append(f"  {step + 1}. {item['name']} ({item['unit']}): {item['reason']}")
        usual = ", ".join(c["name"] for c in plan["paths"]["markov"]["concepts"])
        lines.append(f"What most students did after the same recent topics: {usual}.")
        if plan["review"]:
            lines.append("Topics worth reviewing first: " + ", ".join(r["name"] for r in plan["review"][:3]) + ".")
        return "\n".join(lines)

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
        """DKT rollout of each path (concept ids), as on the Real students page: mean predicted success on the shortlist."""
        dkt, device = self.demo.dkt, self.demo.device
        hist = torch.as_tensor(concepts[-200:], dtype=torch.long, device=device).unsqueeze(0)
        right = torch.as_tensor(correct[-200:], dtype=torch.long, device=device).unsqueeze(0)
        encoded = dkt.encode(hist, right, torch.tensor([hist.size(1)]))
        cands = torch.as_tensor(shortlist, dtype=torch.long, device=device)
        before = float(dkt.probs(encoded)[0][cands].mean())
        generator = torch.Generator(device=device)
        result = {}
        for name, path in paths.items():
            generator.manual_seed(0)
            state = tuple(s.repeat_interleave(rollouts, dim=1) for s in encoded)
            curve = [before]
            for c in path:
                concept = torch.full((rollouts,), int(c), dtype=torch.long, device=device)
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
