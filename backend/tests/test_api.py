import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture(scope="module")
def client(bundle, monkeypatch_module):
    return TestClient(server.create_app(bundle, frontend_dir=None))


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.delenv("GEMINI_API_KEY", raising=False)
    yield mp
    mp.undo()


def test_health_and_info(client):
    assert client.get("/api/health").json()["students"] == 8
    info = client.get("/api/info").json()
    assert info["path_len"] == 3 and "pointer" in info["metrics"] and info["dataset"]["license"] == "CC BY-NC-SA 4.0"


def test_student_and_example_flow(client):
    students = client.get("/api/students").json()
    assert len(students) == 8
    detail = client.get(f"/api/students/{students[0]['id']}").json()
    example_id = detail["examples"][0]["id"]
    ex = client.get(f"/api/examples/{example_id}").json()
    assert ex["student"] == students[0]["id"]
    assert len(ex["candidates"]) == 10 and len(ex["pointer_steps"]) == 3
    assert set(ex["paths"]) == {"pointer", "markov", "popularity", "bkt_gain", "actual"}

    sim = client.get(f"/api/examples/{example_id}/simulate").json()
    assert set(sim["paths"]) == {"pointer", "markov", "popularity", "bkt_gain", "actual"}
    custom = client.post(f"/api/examples/{example_id}/simulate", json={"paths": {"mine": [2, 0]}}).json()
    assert len(custom["paths"]["mine"]["curve"]) == 1 + 2 * sim["attempts_per_concept"]

    graph = client.get(f"/api/examples/{example_id}/graph").json()
    assert {c["id"] for c in ex["candidates"]} <= {n["id"] for n in graph["nodes"]}


def test_errors(client):
    assert client.get("/api/students/999").status_code == 404
    assert client.get("/api/examples/99999").status_code == 404
    bad = client.post("/api/examples/0/simulate", json={"paths": {"bad": [0, 0]}})
    assert bad.status_code == 422
    assert client.post("/api/examples/0/tutor", json={"kind": "concept"}).status_code == 422
    assert client.post("/api/examples/0/tutor", json={"kind": "other"}).status_code == 422


def test_tutor_falls_back_to_a_template_without_a_key(client):
    why = client.post("/api/examples/0/tutor", json={"kind": "why"}).json()
    assert why["source"] == "template" and "1." in why["text"] and "Markov" in why["text"]
    ex = client.get("/api/examples/0").json()
    concept = ex["candidates"][0]["id"]
    text = client.post("/api/examples/0/tutor", json={"kind": "concept", "concept_id": concept}).json()
    assert text["source"] == "template" and ex["candidates"][0]["name"] in text["text"]


def test_tutor_calls_gemini_with_the_model_facts(client, monkeypatch):
    seen = {}

    def fake(api_key, prompt):
        seen.update(api_key=api_key, prompt=prompt)
        return "explained"

    monkeypatch.setattr(server, "ask_gemini", fake)
    out = client.post("/api/examples/0/tutor", json={"kind": "why", "api_key": "k"}).json()
    assert out["text"] == "explained" and seen["api_key"] == "k"
    assert "pointer network chose this path" in seen["prompt"]


def test_serves_the_built_frontend(bundle, tmp_path):
    (tmp_path / "index.html").write_text("<html>pathtrace</html>")
    app = TestClient(server.create_app(bundle, frontend_dir=tmp_path))
    assert "pathtrace" in app.get("/").text
    assert app.get("/api/health").status_code == 200
