import numpy as np
import pytest

from learnpath.catalogue import Shortlister, evaluate_catalogue, shortlist_examples, top_concepts
from learnpath.config import ModelConfig, TrainConfig
from learnpath.preprocess import first_attempts
from learnpath.train import train_model


def test_top_concepts_are_the_highest_scores_best_first():
    scores = np.array([[0.1, 0.9, -np.inf, 0.5, 0.7], [0.3, 0.2, 0.1, 0.0, -np.inf]])
    assert top_concepts(scores, 3).tolist() == [[1, 4, 3], [0, 1, 2]]


def test_shortlists_never_hold_seen_concepts_and_find_the_targets(tiny_processed):
    ex = tiny_processed.examples["test"]
    shortlist, popular, eligible = Shortlister.fit(tiny_processed).rank(tiny_processed.inter, ex, 10, 3, chunk=7)
    for i in range(len(ex)):
        seen = set(tiny_processed.inter.sequence(ex.student[i])[0][: ex.cut[i]].tolist())
        assert not seen & set(shortlist[i].tolist()) and not seen & set(popular[i].tolist())
        assert len(set(shortlist[i].tolist())) == 10
        assert eligible[i] == tiny_processed.inter.num_concepts - len(seen)
    short_ex, found = shortlist_examples(ex, shortlist)
    hit = np.take_along_axis(short_ex.candidates, short_ex.target_slots, axis=1) == ex.targets
    assert np.array_equal(hit, found)


def test_with_the_whole_catalogue_shortlisted_every_target_is_found(tiny_processed, tmp_path):
    cfg = TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu")
    train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "run", log=lambda _: None)
    ex = tiny_processed.examples["test"]
    seen_least = min(len(np.unique(tiny_processed.inter.sequence(s)[0][:c])) for s, c in zip(ex.student, ex.cut))
    everything = tiny_processed.inter.num_concepts - seen_least
    result = evaluate_catalogue(
        tiny_processed, "test", {"pointer": tmp_path / "run" / "model.pt"}, shortlist_size=everything,
        n_boot=20, device="cpu", log=lambda _: None,
    )
    assert result["shortlist"]["target_recall"] == pytest.approx(1.0)
    assert set(result["methods"]) == {"popularity", "markov", "pointer"}
    for values in result["methods"].values():
        assert 0.0 <= values["ndcg"]["mean"] <= 1.0


def test_markov_scorer_matches_the_markov_baseline(tiny_processed):
    from learnpath.baselines import MarkovBaseline

    baseline = MarkovBaseline()
    baseline.fit(tiny_processed)
    shortlister = Shortlister.fit(tiny_processed)
    ex = tiny_processed.examples["val"]
    expected = baseline.score(tiny_processed, "val")
    for i in range(len(ex)):
        scores = shortlister.score_history(tiny_processed.inter.sequence(ex.student[i])[0][: ex.cut[i]])
        # The tiny vocabulary sometimes forces already-seen negatives, which the shortlister rules out.
        unseen = np.isfinite(scores[ex.candidates[i]])
        assert unseen.sum() >= 3
        assert np.allclose(scores[ex.candidates[i]][unseen], expected[i][unseen])


def test_markov_negatives_are_the_shortlist_without_the_targets(tiny_log, tmp_path):
    from learnpath.config import DataConfig
    from learnpath.preprocess import prepare

    proc = prepare(tiny_log, tmp_path / "hard", DataConfig(num_candidates=12, max_windows_per_student=4, negatives="markov"))
    ex = proc.examples["test"]
    shortlister = Shortlister.fit(proc)
    for i in range(len(ex)):
        sequence = proc.inter.sequence(ex.student[i])[0]
        scores = shortlister.score_history(sequence[: ex.cut[i]])
        # Concepts started up to the last target's time window are "next" too, so never negatives.
        distinct, firsts = first_attempts(sequence)
        window = proc.inter.times(ex.student[i])[firsts]
        last_target_window = window[distinct == ex.targets[i][-1]][0]
        scores[distinct[window <= last_target_window]] = -np.inf
        negatives = set(ex.candidates[i].tolist()) - set(ex.targets[i].tolist())
        left_out = [c for c in np.flatnonzero(np.isfinite(scores)) if c not in negatives]
        # every negative scores at least as high as every eligible concept left out
        assert min(scores[list(negatives)]) >= max(scores[left_out], default=-np.inf)
