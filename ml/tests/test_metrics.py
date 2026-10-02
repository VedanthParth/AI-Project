import numpy as np
import pytest

from pathtrace import metrics


def test_chance_matches_paper_closed_forms():
    c = metrics.chance(num_candidates=20, path_len=3)
    assert c["precision"] == pytest.approx(0.15)
    assert c["recall"] == pytest.approx(0.15)
    assert c["ndcg"] == pytest.approx(0.15)
    assert c["hit_rate"] == pytest.approx(1 - 680 / 1140)
    assert c["mrr"] == pytest.approx(3 / 20 + (51 / 380) / 2 + (816 / 6840) / 3)
    assert c["first_step"] == pytest.approx(1 / 20)


@pytest.mark.parametrize("n,k", [(20, 3), (12, 3), (10, 4)])
def test_chance_matches_simulation(n, k):
    rng = np.random.default_rng(0)
    samples = 100_000
    target = np.argsort(rng.random((samples, n)), axis=1)[:, :k]
    pred = np.argsort(rng.random((samples, n)), axis=1)[:, :k]
    observed = metrics.summarize(metrics.per_example(pred, target))
    expected = metrics.chance(n, k)
    for name in metrics.METRICS:
        assert observed[name]["mean"] == pytest.approx(expected[name], abs=0.006), name


def test_perfect_and_reordered_paths():
    target = np.array([[4, 7, 1]])
    perfect = metrics.per_example(target, target)
    assert all(v[0] == pytest.approx(1.0) for v in perfect.values())
    reordered = metrics.per_example(np.array([[1, 4, 7]]), target)
    assert reordered["precision"][0] == 1.0
    assert reordered["ndcg"][0] == pytest.approx(1.0)
    assert reordered["first_step"][0] == 0.0


def test_partial_hit_values():
    values = metrics.per_example(np.array([[9, 4, 8]]), np.array([[4, 7, 1]]))
    assert values["precision"][0] == pytest.approx(1 / 3)
    assert values["mrr"][0] == pytest.approx(1 / 2)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert values["ndcg"][0] == pytest.approx((1 / np.log2(3)) / ideal)


def test_bootstrap_interval_contains_mean():
    values = {"x": np.random.default_rng(1).random(500)}
    summary = metrics.summarize(values, n_boot=500, seed=0)["x"]
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(2)
    labels = rng.random(300) < 0.4
    scores = np.round(rng.random(300) + labels * 0.3, 1)  # rounding creates ties
    pos, neg = scores[labels], scores[~labels]
    pairwise = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert metrics.auc(labels, scores) == pytest.approx(pairwise)


def test_top_k_breaks_ties_at_random():
    picks = metrics.top_k(np.zeros((20_000, 5)), 1, np.random.default_rng(0))[:, 0]
    counts = np.bincount(picks, minlength=5) / len(picks)
    assert np.allclose(counts, 0.2, atol=0.02)
