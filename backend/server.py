"""Maths Learning Path Recommender demo API: serves the trained pointer network on held-out Junyi students.

Run from the repository root (after `pip install -e ml -r backend/requirements.txt`):

    python backend/server.py              # http://localhost:8000

It loads the app bundle (default `backend/bundle`, or $LEARNPATH_BUNDLE) written by
`learnpath export`, and serves the built frontend from `frontend/dist` when it exists.
Set GEMINI_API_KEY (in the environment or `backend/.env`) to let the tutor call Gemini;
without it the tutor answers from a template built from the same numbers.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from learnpath.demo import METHODS, Demo, DemoError
from learnpath.newstudent import NewStudentPlanner

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "bundle"
DEFAULT_FRONTEND = ROOT.parent / "frontend" / "dist"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# Junyi's school stages (Taiwan's grades 1-6, 7-9 and 10-12) named by the Indian classes they roughly match.
CLASSES = {"elementary": "Class 1–6", "junior": "Class 7–9", "senior": "Class 10–12"}


class SimulateRequest(BaseModel):
    paths: dict[str, list[int]] = Field(description="method name -> candidate slots, in order")


class PracticeItem(BaseModel):
    concept: int
    correct: bool


class PlanRequest(BaseModel):
    stage: str = Field(pattern="^(elementary|junior)$")
    strands: dict[int, str] = Field(default_factory=dict, description="strand id -> strong | okay | weak")
    units: dict[int, str] = Field(default_factory=dict, description="unit id -> good | bad")
    current: list[int] = Field(default_factory=list, description="exercises finished last, oldest first")
    practice: list[PracticeItem] = Field(default_factory=list, description="attempts recorded in the app, in order")


class TutorRequest(BaseModel):
    kind: str = Field("why", pattern="^(why|concept)$", description="'why' explains the path, 'concept' teaches one concept")
    concept_id: int | None = None
    api_key: str | None = Field(None, description="optional Gemini key; overrides GEMINI_API_KEY for this request")


def load_env_file(path: Path) -> None:
    """Minimal .env support: KEY=value lines, without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():  # -sig: Notepad may add a BOM
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- tutor


def why_template(demo: Demo, example: int) -> str:
    ex = demo.example(example)
    pointer = ex["paths"]["pointer"]
    lines = []
    for step, slot in enumerate(pointer["slots"]):
        cand = ex["candidates"][slot]
        prob = ex["pointer_steps"][step][slot]
        earlier = {ex["candidates"][s]["id"] for s in pointer["slots"][:step]}
        reason = f"BKT puts the student's mastery at {cand['mastery']:.0%}"
        covered = [p["name"] for p in cand["unmet_prereqs"] if p["id"] in earlier]
        missing = [p["name"] for p in cand["unmet_prereqs"] if p["id"] not in earlier]
        if covered:
            reason += f", and it comes after its prerequisite {covered[0]} earlier in the path"
        if missing:
            reason += f", though its inferred prerequisite {missing[0]} isn't done yet"
        lines.append(f"{step + 1}. {cand['name']}: the model gave it {prob:.0%} of its attention at this step; {reason}.")
    markov = ", ".join(c["name"] for c in ex["paths"]["markov"]["concepts"])
    lines.append(f"For comparison, the Markov baseline, which only looks at the last few concepts, would pick: {markov}.")
    return "\n".join(lines)


def concept_template(demo: Demo, concept_id: int) -> str:
    c = demo.concepts[concept_id]
    parents = [demo.concepts[p]["name"] for p in demo.parents[concept_id]]
    children = [demo.concepts[d]["name"] for d in demo.children[concept_id]]
    classes = f" It is taught in {CLASSES[c['stage']]}." if c.get("stage") in CLASSES else ""
    text = f"{c['name']} is a maths exercise from Junyi Academy.{classes}"
    if parents:
        text += " Students usually practise it after: " + ", ".join(parents[:5]) + "."
    if children:
        text += " It usually leads on to: " + ", ".join(children[:5]) + "."
    return text + " Add a Gemini API key for a full explanation."


def ask_gemini(api_key: str, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text


def tutor_prompt(demo: Demo, example: int, req: TutorRequest) -> str:
    if req.kind == "why":
        return (
            "You are a maths teacher in an Indian school explaining an AI system's recommendation to a student and "
            "their parents. Using only the facts below, explain in at most 4 short sentences why this 3-topic path "
            "suits the student now, and how it differs from the baseline. Don't invent numbers. Write in Indian "
            "English.\n\n"
            + demo.tutor_context(example)
        )
    c = demo.concepts[req.concept_id]
    parents = ", ".join(demo.concepts[p]["name"] for p in demo.parents[req.concept_id]) or "none recorded"
    return (
        f"You are a friendly maths teacher in an Indian school. Explain the topic \"{c['name']}\" "
        f"({CLASSES.get(c.get('stage'), 'school level')}) in 3 short paragraphs for a student about to practise it, "
        "with one worked example set in an everyday Indian context, such as rupees at a shop, a cricket score or a "
        f"train journey. Use Indian English spelling and metric units. Topics students usually learn first: {parents}."
    )


# --------------------------------------------------------------------------- app


def create_app(bundle_dir: str | Path = DEFAULT_BUNDLE, frontend_dir: str | Path | None = DEFAULT_FRONTEND) -> FastAPI:
    demo = Demo(bundle_dir)
    app = FastAPI(title="Maths Learning Path Recommender demo API", version="1.0")
    example = lru_cache(maxsize=512)(demo.example)
    students = demo.students()
    try:
        planner = NewStudentPlanner(demo, bundle_dir)
    except DemoError:
        planner = None  # an older bundle without topics.json: no New student page
    results_dir = Path(bundle_dir) / "results"
    results = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted(results_dir.glob("*.json"))} if results_dir.is_dir() else {}

    def guard(fn, *args):
        try:
            return fn(*args)
        except DemoError as err:
            raise HTTPException(status_code=404 if "unknown" in str(err) else 422, detail=str(err)) from err

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "students": demo.inter.num_students, "examples": len(demo.ex)}

    @app.get("/api/info")
    def info() -> dict:
        return demo.info()

    @app.get("/api/students")
    def list_students() -> list[dict]:
        return students

    @app.get("/api/students/{student_id}")
    def get_student(student_id: int) -> dict:
        return guard(demo.student, student_id)

    @app.get("/api/examples/{example_id}")
    def get_example(example_id: int) -> dict:
        return guard(example, example_id)

    @app.get("/api/examples/{example_id}/simulate")
    def simulate_methods(example_id: int) -> dict:
        ex = guard(example, example_id)
        paths = {name: ex["paths"][name]["slots"] for name in (*METHODS, "actual")}
        return guard(demo.simulate, example_id, paths)

    @app.post("/api/examples/{example_id}/simulate")
    def simulate_paths(example_id: int, req: SimulateRequest) -> dict:
        return guard(demo.simulate, example_id, req.paths)

    @app.get("/api/questionnaire")
    def questionnaire() -> dict:
        if planner is None:
            raise HTTPException(status_code=404, detail="this bundle has no topics.json; re-export it with --content")
        return planner.questionnaire()

    @app.post("/api/plan")
    def plan(req: PlanRequest) -> dict:
        if planner is None:
            raise HTTPException(status_code=404, detail="this bundle has no topics.json; re-export it with --content")
        practice = [(p.concept, p.correct) for p in req.practice]
        return guard(planner.plan, req.stage, req.strands, req.units, req.current, practice)

    @app.get("/api/results")
    def get_results() -> dict:
        return results

    @app.get("/api/examples/{example_id}/graph")
    def graph(example_id: int) -> dict:
        return guard(demo.neighbourhood, example_id)

    @app.post("/api/examples/{example_id}/tutor")
    def tutor(example_id: int, req: TutorRequest) -> dict:
        guard(example, example_id)
        if req.kind == "concept" and req.concept_id not in demo.concepts:
            raise HTTPException(status_code=422, detail="concept_id must be a known concept for kind 'concept'")
        key = req.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            text = why_template(demo, example_id) if req.kind == "why" else concept_template(demo, req.concept_id)
            return {"source": "template", "text": text}
        try:
            return {"source": GEMINI_MODEL, "text": ask_gemini(key, tutor_prompt(demo, example_id, req))}
        except Exception as err:  # the SDK raises several error types; report them all the same way
            raise HTTPException(status_code=502, detail=f"Gemini request failed: {err}") from err

    if frontend_dir and Path(frontend_dir).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


def main() -> None:
    import uvicorn

    load_env_file(ROOT / ".env")
    bundle = os.environ.get("LEARNPATH_BUNDLE", str(DEFAULT_BUNDLE))
    port = int(os.environ.get("PORT", "8000"))
    print(f"Maths Learning Path Recommender demo: bundle {bundle}, http://localhost:{port}")
    uvicorn.run(create_app(bundle), host=os.environ.get("HOST", "127.0.0.1"), port=port)


if __name__ == "__main__":
    main()
