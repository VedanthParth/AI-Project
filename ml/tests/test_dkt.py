import numpy as np
import pytest
import torch

from pathtrace.dkt import DKT, DKTConfig, evaluate_dkt, learning_gain, train_dkt


@pytest.fixture(scope="module")
def trained(tiny_processed, tmp_path_factory):
    cfg = DKTConfig(hidden_dim=32, embed_dim=16, max_epochs=3, patience=3, batch_size=32)
    out = tmp_path_factory.mktemp("dkt")
    summary = train_dkt(tiny_processed, cfg, out, log=lambda _: None)
    from pathtrace.dkt import load_dkt

    return load_dkt(out / "dkt.pt", torch.device("cpu")), summary


def test_dkt_learns_something(trained):
    _, summary = trained
    assert summary["test"]["auc"] > 0.6


def test_step_continues_the_encoded_history(tiny_processed):
    torch.manual_seed(0)
    model = DKT(tiny_processed.inter.num_concepts, DKTConfig(hidden_dim=16, embed_dim=8)).eval()
    concepts = torch.tensor([[3, 5, 3, 7]])
    correct = torch.tensor([[1, 0, 1, 1]])
    full = model.encode(concepts, correct, torch.tensor([4]))
    stepped = model.step(concepts[:, 3], correct[:, 3].float(), model.encode(concepts[:, :3], correct[:, :3], torch.tensor([3])))
    assert torch.allclose(full[0], stepped[0], atol=1e-6)
    assert torch.allclose(full[1], stepped[1], atol=1e-6)


def test_learning_gain_uses_common_random_numbers(trained, tiny_processed):
    model, _ = trained
    ex = tiny_processed.examples["test"]
    rng = np.random.default_rng(0)
    paths = np.argsort(rng.random(ex.candidates.shape), axis=1)[:, :3]
    out = learning_gain(model, tiny_processed.inter, ex, {"a": paths, "b": paths.copy()}, rollouts=2, max_history=50)
    assert np.array_equal(out["a"]["after"], out["b"]["after"])
    r = out["a"]
    assert np.allclose(r["gain"], (r["after"] - r["before"]) / (1 - r["before"]))
    assert ((r["before"] > 0) & (r["before"] < 1)).all()


def test_evaluate_dkt_reports_auc(trained, tiny_processed):
    model, _ = trained
    result = evaluate_dkt(model, tiny_processed.inter, tiny_processed.splits["val"], DKTConfig(), torch.device("cpu"))
    assert 0.5 < result["auc"] <= 1.0 and result["predictions"] > 0
