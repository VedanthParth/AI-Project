from itertools import permutations

import numpy as np
import pytest
import torch

from pathtrace import bkt, prereq
from pathtrace.batching import make_batch
from pathtrace.config import ModelConfig, TrainConfig
from pathtrace.rl import RewardTables, RLConfig, finetune, oracle_paths
from pathtrace.train import train_model


def _tables(seed=0, examples=40, n=6, weight=0.2):
    rng = np.random.default_rng(seed)
    need = rng.random((examples, n, n)) < 0.15
    need[:, np.arange(n), np.arange(n)] = False
    return RewardTables(
        gains=rng.random((examples, n)), always=rng.random((examples, n)) < 0.2, need=need, violation_weight=weight
    )


def test_reward_is_gain_minus_weighted_violations():
    tables = _tables()
    idx = np.arange(len(tables.gains))
    paths = np.tile(np.array([2, 0, 5]), (len(idx), 1))
    expected = bkt.path_reward(tables.gains, paths) - 0.2 * prereq.path_violations(paths, tables.always, tables.need)
    assert np.allclose(tables.reward(idx, paths), expected)


@pytest.mark.parametrize("weight", [0.0, 0.2, 5.0])
def test_oracle_is_exact(weight):
    tables = _tables(weight=weight)
    idx = np.arange(len(tables.gains))
    best = tables.reward(idx, oracle_paths(tables, 3))
    all_paths = np.array(list(permutations(range(6), 3)))
    brute = np.max([tables.reward(idx, np.tile(p, (len(idx), 1))) for p in all_paths], axis=0)
    assert np.allclose(best, brute)


@pytest.fixture(scope="module")
def setup(tiny_processed, tmp_path_factory):
    out = tmp_path_factory.mktemp("rl")
    cfg = TrainConfig(max_epochs=2, bootstrap=0, max_history=30, batch_size=32)
    train_model(tiny_processed, ModelConfig(), cfg, out / "sup", log=lambda _: None)
    params, _ = bkt.fit_bkt(tiny_processed.inter, tiny_processed.splits["train"], min_attempts=10)
    graph, _ = prereq.infer_graph(tiny_processed.inter, tiny_processed.splits["train"], min_support=5)
    return out, params, graph


@pytest.mark.parametrize("baseline", ["greedy", "ema"])
def test_finetune_runs_and_never_returns_a_worse_policy(tiny_processed, setup, baseline):
    out, params, graph = setup
    cfg = RLConfig(max_epochs=3, patience=3, lr=2e-3, bootstrap=0, max_history=30, baseline=baseline)
    results = finetune(tiny_processed, out / "sup" / "model.pt", params, graph, cfg, out / baseline, log=lambda _: None)
    start = results["start"]["reward"]["mean"]
    assert results["splits"]["val"]["reward"]["mean"] >= start - 1e-9
    assert 0 < results["splits"]["test"]["share_of_oracle"] <= 1.0 + 1e-9
    assert (out / baseline / "preds_test.npy").exists()


def test_positive_advantage_raises_the_sampled_paths_probability(tiny_processed, setup):
    from pathtrace.train import load_checkpoint

    out, _, _ = setup
    model, _ = load_checkpoint(out / "sup" / "model.pt", torch.device("cpu"))
    model.eval()  # no dropout, so the before/after comparison is deterministic
    batch = make_batch(tiny_processed.inter, tiny_processed.examples["train"], np.arange(16), 30)
    torch.manual_seed(0)
    sample = model.recommend(batch, 3, mode="sample")
    actions = sample.actions

    def path_log_prob():
        out = model.decode(model.encode_student(batch), model.encode_candidates(batch.candidates), 3, "teacher", actions)
        return out.log_probs.sum(dim=1)

    before = path_log_prob().detach()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss = -(torch.ones(16) * path_log_prob()).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert (path_log_prob().detach() - before).mean() > 0
