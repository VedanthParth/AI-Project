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
from collections.abc import Callable
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


class PlanExplainRequest(PlanRequest):
    kind: str = Field("why", pattern="^(why|concept)$", description="'why' explains the plan, 'concept' teaches one topic")
    concept_id: int | None = None
    style: str = Field("pointer", pattern="^(pointer|rl)$", description="which recommendation the page shows")


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

STYLE = (
    "Write in Indian English, in plain words a student and their parents understand, and only in English: no "
    "Hindi or other words, greetings or nicknames. Never use a name, a placeholder such as [Name], or 'he' or 'she'. Don't mention models, algorithms, probabilities or any other "
    "technical term, and don't invent numbers or facts that aren't given."
)


def plan_template(items: list[dict], usual: list[dict]) -> str:
    """The tutor's answer without Gemini: each topic with the reason the app already shows."""
    lines = [f"{step + 1}. {item['name']}: {item['reason']}" for step, item in enumerate(items)]
    lines.append("Most students who had done the same recent topics went on to: " + ", ".join(c["name"] for c in usual) + ".")
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
    return text + " A fuller explanation with a worked example appears when the tutor is connected to Gemini."


def ask_gemini(api_key: str, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text


def why_prompt(facts: str, about: str) -> str:
    """``about`` is "you" for a student reading their own plan, "this student" for an example student."""
    audience = (
        "Talk to the student as 'you'."
        if about == "you"
        else "You are describing an example student to a parent: call them 'this student'."
    )
    return (
        "You are a friendly maths teacher in an Indian school. Using only the facts below, explain in 3 or 4 short "
        f"sentences why these three topics, in this order, suit the student now. {audience} {STYLE}\n\n{facts}"
    )


def topic_prompt(demo: Demo, concept_id: int) -> str:
    c = demo.concepts[concept_id]
    parents = ", ".join(demo.concepts[p]["name"] for p in demo.parents[concept_id]) or "none recorded"
    return (
        f"You are a friendly maths teacher in an Indian school. Explain the topic \"{c['name']}\" "
        f"({CLASSES.get(c.get('stage'), 'school level')}) in 3 short paragraphs for a student about to practise it, "
        "with one worked example set in an everyday Indian context, such as rupees at a shop, a cricket score or a "
        f"train journey. Talk to the student as 'you' and use metric units. {STYLE} "
        f"Topics students usually learn first: {parents}."
    )


def answer(key: str | None, prompt: Callable[[], str], template: Callable[[], str]) -> dict:
    """Gemini's answer when a key is set, otherwise the template; ``source`` says which."""
    if not key:
        return {"source": "template", "text": template()}
    try:
        return {"source": "gemini", "model": GEMINI_MODEL, "text": ask_gemini(key, prompt())}
    except Exception as err:  # the SDK raises several error types; report them all the same way
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {err}") from err


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
        # Whether the tutor answers with Gemini (a key in backend/.env) or from templates; never the key itself.
        return {**demo.info(), "tutor": "gemini" if os.environ.get("GEMINI_API_KEY") else "template"}

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
        ex = guard(example, example_id)
        if req.kind == "concept" and req.concept_id not in demo.concepts:
            raise HTTPException(status_code=422, detail="concept_id must be a known concept for kind 'concept'")
        key = req.api_key or os.environ.get("GEMINI_API_KEY")
        if req.kind == "concept":
            return answer(key, lambda: topic_prompt(demo, req.concept_id), lambda: concept_template(demo, req.concept_id))
        return answer(
            key,
            lambda: why_prompt(demo.tutor_context(example_id), "this student"),
            lambda: plan_template(ex["paths"]["pointer"]["concepts"], ex["paths"]["markov"]["concepts"]),
        )

    @app.post("/api/plan/explain")
    def explain_plan(req: PlanExplainRequest) -> dict:
        if planner is None:
            raise HTTPException(status_code=404, detail="this bundle has no topics.json; re-export it with --content")
        if req.kind == "concept" and req.concept_id not in demo.concepts:
            raise HTTPException(status_code=422, detail="concept_id must be a known concept for kind 'concept'")
        key = os.environ.get("GEMINI_API_KEY")
        if req.kind == "concept":
            return answer(key, lambda: topic_prompt(demo, req.concept_id), lambda: concept_template(demo, req.concept_id))
        practice = [(p.concept, p.correct) for p in req.practice]
        plan = guard(planner.plan, req.stage, req.strands, req.units, req.current, practice)
        style = req.style if req.style in plan["paths"] else "pointer"
        return answer(
            key,
            lambda: why_prompt(planner.facts(plan, style), "you"),
            lambda: plan_template(plan["paths"][style]["concepts"], plan["paths"]["markov"]["concepts"]),
        )

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
