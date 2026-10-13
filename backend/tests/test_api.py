import pytest
import server
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The API on the committed bundle, serving a stand-in built frontend, with no Gemini key in the environment."""
    frontend = tmp_path_factory.mktemp("dist")
    (frontend / "index.html").write_text("<html>pathtrace</html>", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("GEMINI_API_KEY", raising=False)
        yield TestClient(server.create_app(server.DEFAULT_BUNDLE, frontend_dir=frontend))


def test_info_and_students(client):
    info = client.get("/api/info").json()
    assert info["students"] == 300 and info["path_len"] == 3 and "pointer" in info["metrics"]
    assert info["dataset"]["license"] == "CC BY-NC-SA 4.0"
    students = client.get("/api/students").json()
    assert len(students) == 300 and all(s["examples"] > 0 for s in students)


def test_example_flow(client):
    student = client.get("/api/students").json()[0]["id"]
    detail = client.get(f"/api/students/{student}").json()
    example_id = detail["examples"][0]["id"]
    ex = client.get(f"/api/examples/{example_id}").json()
    assert ex["student"] == student
    assert len(ex["candidates"]) == 20 and len(ex["pointer_steps"]) == 3
    assert set(ex["paths"]) == {"pointer", "markov", "popularity", "bkt_gain", "actual"}

    sim = client.get(f"/api/examples/{example_id}/simulate").json()
    assert set(sim["paths"]) == set(ex["paths"])
    assert all(len(p["curve"]) == 1 + 3 * sim["attempts_per_concept"] for p in sim["paths"].values())

    graph = client.get(f"/api/examples/{example_id}/graph").json()
    assert {c["id"] for c in ex["candidates"]} <= {n["id"] for n in graph["nodes"]}


def test_errors(client):
    assert client.get("/api/students/999").status_code == 404
    assert client.get("/api/examples/99999").status_code == 404
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


def test_serves_the_built_frontend_beside_the_api(client):
    assert "pathtrace" in client.get("/").text
    assert client.get("/api/info").status_code == 200
