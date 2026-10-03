import numpy as np
import pandas as pd
import pytest
import torch

from pathtrace.baselines import MarkovBaseline
from pathtrace.batching import make_batch
from pathtrace.config import DataConfig, ModelConfig, TrainConfig
from pathtrace.demo import Demo, DemoError
from pathtrace.dkt import DKTConfig
from pathtrace.export import export_bundle
from pathtrace.grid import run_grid
from pathtrace.preprocess import load_processed, prepare
from pathtrace.train import load_checkpoint


@pytest.fixture(scope="module")
def bundle(tiny_log, tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    data = root / "processed"
    prepare(tiny_log, data, DataConfig(num_candidates=12, max_windows_per_student=4))
    run_grid(
        data,
        root / "runs",
        seeds=[0],
        variants=["pointer"],
        model_cfg=ModelConfig(embed_dim=16, hidden_dim=16, ff_dim=32),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32),
        dkt_cfg=DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=1),
        rl=False,
        gain=False,
        log=lambda _: None,
    )
    proc = load_processed(data)
    names = root / "names.csv"
    ucids = proc.inter.concept_names[1:4]
    pd.DataFrame({"ucid": ucids, "name_en": ["Alpha", "Beta", "Gamma"], "level": ["basic", "", "advanced"]}).to_csv(
        names, index=False
    )
    out = root / "bundle"
    meta = export_bundle(data, root / "runs", out, names_csv=names, max_students=10, log=lambda _: None)
    return proc, out, meta, Demo(out)


def test_export_keeps_only_sampled_test_students(bundle):
    proc, out, meta, demo = bundle
    assert meta["students"] == demo.inter.num_students == 10
    test_students = set(proc.splits["test"].tolist())
    sub = demo.proc
    assert len(sub.examples["train"]) == len(sub.examples["val"]) == 0
    # Each bundled student's attempts match some test student's, and examples carry over unchanged.
    original = {}
    for s in test_students:
        concepts, correct = proc.inter.sequence(s)
        original[(concepts.tobytes(), correct.tobytes())] = s
    for s in range(sub.inter.num_students):
        concepts, correct = sub.inter.sequence(s)
        src = original[(concepts.tobytes(), correct.tobytes())]
        ours = sub.examples["test"].cut[sub.examples["test"].student == s]
        theirs = proc.examples["test"].cut[proc.examples["test"].student == src]
        assert np.array_equal(np.sort(ours), np.sort(theirs))
    assert meta["metrics"]["pointer"]["ndcg"]["mean"] >= 0
    assert (out / "README.md").exists()


def test_concept_names_come_from_the_csv_with_ucid_fallback(bundle):
    _, _, _, demo = bundle
    assert [demo.concepts[i]["name"] for i in (1, 2, 3)] == ["Alpha", "Beta", "Gamma"]
    assert demo.concepts[3]["level"] == "advanced"
    assert demo.concepts[4]["name"] == str(demo.inter.concept_names[4])


def test_markov_from_saved_counts_matches_a_fresh_fit(bundle):
    proc = bundle[0]
    fresh = MarkovBaseline()
    fresh.fit(proc)
    with np.load(bundle[1] / "markov.npz") as z:
        saved = MarkovBaseline()
        saved.set_counts(z["src"], z["dst"], z["counts"], proc.popularity)
    assert np.allclose(fresh.transition, saved.transition)


def test_example_paths_and_pointer_attention(bundle):
    proc, out, _, demo = bundle
    e = 0
    ex = demo.example(e)
    k, n = demo.k, demo.ex.candidates.shape[1]
    assert set(ex["paths"]) == {"pointer", "markov", "popularity", "bkt_gain", "actual"}
    assert ex["paths"]["actual"]["hits"] == k and ex["paths"]["actual"]["first_step"]
    for path in ex["paths"].values():
        assert len(set(path["slots"])) == k and all(0 <= s < n for s in path["slots"])
    steps = np.array(ex["pointer_steps"])
    assert steps.shape == (k, n) and np.allclose(steps.sum(axis=1), 1.0, atol=1e-3)
    pointer = ex["paths"]["pointer"]["slots"]
    for t in range(1, k):  # earlier picks are masked out at later steps
        assert steps[t, pointer[:t]].max() < 1e-6
    assert sorted(c["target_rank"] for c in ex["candidates"] if c["target_rank"]) == list(range(1, k + 1))

    # The bundled checkpoint gives the same greedy path as on the source data.
    model, ckpt = load_checkpoint(out / "model.pt", torch.device("cpu"))
    batch = make_batch(demo.inter, demo.ex, np.array([e]), ckpt["train_config"]["max_history"])
    with torch.no_grad():
        assert model.recommend(batch, k).actions[0].tolist() == pointer


def test_simulation_is_repeatable_and_shaped(bundle):
    _, _, _, demo = bundle
    ex = demo.example(1)
    paths = {name: ex["paths"][name]["slots"] for name in ("pointer", "markov")}
    first = demo.simulate(1, paths, rollouts=8)
    again = demo.simulate(1, paths, rollouts=8)
    assert first == again
    for result in first["paths"].values():
        assert len(result["curve"]) == 1 + 3 * demo.k
        assert result["curve"][0] == result["before"] and result["curve"][-1] == result["after"]
    single = demo.simulate(1, {"mine": [0]}, rollouts=8)["paths"]["mine"]
    assert len(single["curve"]) == 4


@pytest.mark.parametrize("path", [[], [0, 0], [0, 1, 2, 3], [99]])
def test_invalid_paths_are_rejected(bundle, path):
    _, _, _, demo = bundle
    with pytest.raises(DemoError):
        demo.simulate(0, {"bad": path})


def test_unknown_ids_are_rejected(bundle):
    _, _, _, demo = bundle
    with pytest.raises(DemoError):
        demo.example(len(demo.ex))
    with pytest.raises(DemoError):
        demo.student(-1)


def test_students_and_neighbourhood(bundle):
    _, _, _, demo = bundle
    students = demo.students()
    assert len(students) == 10 and all(s["examples"] > 0 for s in students)
    detail = demo.student(0)
    cuts = [e["cut"] for e in detail["examples"]]
    assert cuts == sorted(cuts) and len(detail["attempts"]) == students[0]["attempts"]

    graph = demo.neighbourhood(0)
    ids = {node["id"] for node in graph["nodes"]}
    assert set(demo.ex.candidates[0].tolist()) <= ids
    assert all(edge["source"] in ids and edge["target"] in ids for edge in graph["edges"])
    assert demo.example(0)["paths"]["pointer"]["concepts"][0]["name"] in demo.tutor_context(0)
