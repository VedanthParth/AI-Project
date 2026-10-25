import pytest

from learnpath.config import DataConfig, ModelConfig, TrainConfig
from learnpath.demo import Demo, DemoError
from learnpath.dkt import DKTConfig
from learnpath.export import export_bundle
from learnpath.grid import run_grid
from learnpath.newstudent import REVIEW_BELOW, NewStudentPlanner
from learnpath.preprocess import prepare
from learnpath.utils import load_json, save_json


@pytest.fixture(scope="module")
def planner(tiny_log, tmp_path_factory):
    """A bundle with an RL policy and a made-up topic hierarchy: 2 junior strands, 6 units (concept id mod 6)."""
    root = tmp_path_factory.mktemp("newstudent")
    prepare(tiny_log, root / "data", DataConfig(num_candidates=12, max_windows_per_student=4))
    run_grid(
        root / "data", root / "runs", seeds=[0], variants=["pointer"], sensitivity=False, gain=False,
        model_cfg=ModelConfig(embed_dim=16, hidden_dim=16, ff_dim=32),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu"),
        dkt_cfg=DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=1, device="cpu"), log=lambda _: None,
    )
    names = root / "names.csv"
    names.write_text("ucid,name_en,level\n", encoding="utf-8")
    out = root / "bundle"
    export_bundle(root / "data", root / "runs", out, names_csv=names, max_students=10, log=lambda _: None)
    concepts = load_json(out / "concepts.json")
    for c in concepts:
        c["stage"] = "junior"
    save_json(concepts, out / "concepts.json")
    save_json(
        {
            "strands": [{"id": j, "name": f"Strand {j}", "stage": "junior"} for j in range(2)],
            "units": [
                {"id": i, "strand": i % 2, "name": f"Unit {i}", "stage": "junior",
                 "concepts": [c["id"] for c in concepts if c["id"] % 6 == i]}
                for i in range(6)
            ],
        },
        out / "topics.json",
    )
    demo = Demo(out)
    return NewStudentPlanner(demo, out)


def test_questionnaire_lists_units_with_exercises_in_curriculum_order(planner):
    q = planner.questionnaire()
    assert len(q["strands"]) == 2 and len(q["units"]) == 6
    for unit in q["units"]:
        positions = [planner.position[e["id"]] for e in unit["exercises"]]
        assert positions == sorted(positions)


def test_plan_recommends_unseen_stage_concepts_and_explains_each_step(planner):
    unit0 = planner.questionnaire()["units"][0]["exercises"]
    plan = planner.plan("junior", {1: "strong"}, {0: "bad"}, [unit0[2]["id"]], [])
    seen = {h["id"] for h in plan["history"]}
    shortlist = [s["id"] for s in plan["shortlist"]]
    assert not seen & set(shortlist) and len(shortlist) == plan["pipeline"]["shortlist_size"]
    assert {"markov", "pointer", "rl"} <= set(plan["paths"])
    assert {"markov", "pointer", "rl", "popularity"} == set(plan["paths"])
    for name, path in plan["paths"].items():
        assert len({c["id"] for c in path["concepts"]}) == 3
        assert not seen & {c["id"] for c in path["concepts"]}
        if name != "popularity":  # the most common topics come from every option, not the shortlist
            assert all(c["id"] in shortlist for c in path["concepts"])
    steps = plan["paths"]["pointer"]["steps"]
    assert len(steps) == 3 and all(abs(sum(row) - 1) < 1e-3 for row in steps)
    sources = {h["source"] for h in plan["history"]}
    assert "you finished this last" in sources and "you marked Unit 0 as needs help" in sources
    assert set(plan["simulation"]["paths"]) == set(plan["paths"])


def test_struggling_in_practice_lowers_mastery_and_lists_it_for_review(planner):
    unit0 = planner.questionnaire()["units"][0]["exercises"]
    first = planner.plan("junior", {}, {}, [unit0[0]["id"]], [])
    step1 = first["paths"]["pointer"]["concepts"][0]["id"]
    after = planner.plan("junior", {}, {}, [unit0[0]["id"]], [(step1, False)] * 3)
    entry = next(h for h in after["history"] if h["id"] == step1)
    assert entry["source"] == "practised here" and entry["correct"] == 0
    assert step1 not in [c["id"] for c in after["paths"]["pointer"]["concepts"]]
    assert entry["mastery"] >= REVIEW_BELOW or step1 in [r["id"] for r in after["review"]]


def test_bad_requests_are_rejected(planner):
    with pytest.raises(DemoError):
        planner.plan("senior", {}, {}, [1], [])
    with pytest.raises(DemoError):
        planner.plan("junior", {}, {}, [], [])
    with pytest.raises(DemoError):
        planner.plan("junior", {0: "brilliant"}, {}, [1], [])
    with pytest.raises(DemoError):
        planner.plan("junior", {}, {}, [1, 2, 3, 4], [])


def test_every_explanation_score_is_between_0_and_1(planner):
    unit0 = planner.questionnaire()["units"][0]["exercises"]
    plan = planner.plan("junior", {1: "strong"}, {0: "bad"}, [unit0[2]["id"]], [])
    for name, path in plan["paths"].items():
        assert set(path["profile"]) == {"ready", "room", "recent", "level", "gain", "progress"}
        assert all(0 <= v <= 1 for v in path["profile"].values())
        for item in path["concepts"]:
            factors = item["factors"]
            assert all(0 <= factors[k] <= 1 for k in ("ready", "room", "recent", "level", "gain"))
            assert (factors["sure"] is None) == (name in {"markov", "popularity"})
            assert item["reason"] and "Markov" not in item["reason"]
    assert max(p["profile"]["progress"] for p in plan["paths"].values()) == 1.0
    assert max(p["profile"]["gain"] for p in plan["paths"].values()) == 1.0


def test_strengths_and_practice_trace_follow_the_answers(planner):
    unit0 = planner.questionnaire()["units"][0]["exercises"]
    plan = planner.plan("junior", {1: "strong"}, {}, [unit0[1]["id"]], [])
    assert [a["name"] for a in plan["areas"]["now"]] == ["Strand 0", "Strand 1"]
    assert all(a["known"] for a in plan["areas"]["now"])  # strand 1 is strong; unit 0 (strand 0) holds the named exercise
    assert plan["practice_trace"] == []
    step1 = plan["paths"]["pointer"]["concepts"][0]["id"]
    after = planner.plan("junior", {1: "strong"}, {}, [unit0[1]["id"]], [(step1, False), (step1, False)])
    trace = next(t for t in after["practice_trace"] if t["id"] == step1)
    assert len(trace["points"]) == 3 and trace["points"][0] == next(c for c in plan["shortlist"] if c["id"] == step1)["mastery"]
    assert after["areas"]["before"] == plan["areas"]["now"]
    got_it = planner.plan("junior", {1: "strong"}, {}, [unit0[1]["id"]], [(step1, True), (step1, True)])
    assert next(t for t in got_it["practice_trace"] if t["id"] == step1)["points"][-1] > trace["points"][-1]


def test_tutor_facts_are_plain_and_name_the_plan(planner):
    unit0 = planner.questionnaire()["units"][0]["exercises"]
    plan = planner.plan("junior", {}, {0: "good"}, [unit0[1]["id"]], [])
    facts = planner.facts(plan)
    assert "Unit 0" in facts and plan["paths"]["pointer"]["concepts"][0]["name"] in facts
    assert not any(term in facts for term in ("Markov", "BKT", "pointer", "probability"))
