import shutil

import pytest
from fastapi.testclient import TestClient

from learnpath.utils import load_json, save_json
from server import create_app


@pytest.fixture(scope="module")
def topics_client(bundle, tmp_path_factory):
    """The test bundle plus a made-up topic hierarchy: 2 junior strands, 4 units (concept id mod 4)."""
    out = tmp_path_factory.mktemp("topics") / "bundle"
    shutil.copytree(bundle, out)
    concepts = load_json(out / "concepts.json")
    for c in concepts:
        c["stage"] = "junior"
    save_json(concepts, out / "concepts.json")
    save_json(
        {
            "strands": [{"id": j, "name": f"Strand {j}", "stage": "junior"} for j in range(2)],
            "units": [
                {"id": i, "strand": i % 2, "name": f"Unit {i}", "stage": "junior",
                 "concepts": [c["id"] for c in concepts if c["id"] % 4 == i]}
                for i in range(4)
            ],
        },
        out / "topics.json",
    )
    (out / "results").mkdir(exist_ok=True)
    save_json({"methods": {}}, out / "results" / "catalogue_test_20.json")
    return TestClient(create_app(out, frontend_dir=None))


def test_older_bundles_have_no_new_student_page(bundle):
    client = TestClient(create_app(bundle, frontend_dir=None))
    assert client.get("/api/questionnaire").status_code == 404
    assert client.get("/api/results").json() == {}


def test_questionnaire_then_plan_then_practice(topics_client):
    q = topics_client.get("/api/questionnaire").json()
    assert len(q["strands"]) == 2 and len(q["units"]) == 4
    last = q["units"][0]["exercises"][1]["id"]
    body = {"stage": "junior", "strands": {"1": "strong"}, "units": {"0": "bad"}, "current": [last]}
    plan = topics_client.post("/api/plan", json=body).json()
    step1 = plan["paths"]["pointer"]["concepts"][0]["id"]
    assert len(plan["paths"]["pointer"]["concepts"]) == 3

    body["practice"] = [{"concept": step1, "correct": False}]
    after = topics_client.post("/api/plan", json=body).json()
    assert step1 not in [c["id"] for c in after["paths"]["pointer"]["concepts"]]
    assert after["pipeline"]["history_attempts"] == plan["pipeline"]["history_attempts"] + 1


def test_plan_rejects_bad_answers(topics_client):
    assert topics_client.post("/api/plan", json={"stage": "senior", "current": [1]}).status_code == 422
    assert topics_client.post("/api/plan", json={"stage": "junior"}).status_code == 422
    assert topics_client.post("/api/plan", json={"stage": "junior", "strands": {"0": "great"}, "current": [1]}).status_code == 422


def test_results_come_from_the_bundle(topics_client):
    assert topics_client.get("/api/results").json() == {"catalogue_test_20": {"methods": {}}}


def test_info_says_whether_the_tutor_uses_gemini(topics_client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert topics_client.get("/api/info").json()["tutor"] == "template"
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    response = topics_client.get("/api/info")
    assert response.json()["tutor"] == "gemini" and "secret-test-key" not in response.text


def test_plan_explanations_with_and_without_gemini(topics_client, monkeypatch):
    import server

    q = topics_client.get("/api/questionnaire").json()
    body = {"stage": "junior", "units": {"0": "bad"}, "current": [q["units"][0]["exercises"][1]["id"]]}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    plain = topics_client.post("/api/plan/explain", json=body).json()
    assert plain["source"] == "template" and plain["text"].startswith("1. ")

    seen = {}
    monkeypatch.setattr(server, "ask_gemini", lambda key, prompt: seen.update(key=key, prompt=prompt) or "explained")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    out = topics_client.post("/api/plan/explain", json=body).json()
    assert out == {"source": "gemini", "model": server.GEMINI_MODEL, "text": "explained"}
    assert "'you'" in seen["prompt"] and "Never use a name" in seen["prompt"] and "Markov" not in seen["prompt"]
    topic = topics_client.post("/api/plan/explain", json={**body, "kind": "concept", "concept_id": 1}).json()
    assert topic["text"] == "explained" and "worked example" in seen["prompt"]
    assert topics_client.post("/api/plan/explain", json={**body, "kind": "concept", "concept_id": 99999}).status_code == 422
