import numpy as np
import pandas as pd

from pathtrace.config import DataConfig
from pathtrace.preprocess import (
    SPLITS,
    build_examples,
    concept_popularity,
    first_attempts,
    load_log,
    load_processed,
    prepare,
)


def test_load_log_sorts_by_time_and_parses_correctness(tmp_path):
    rows = [
        ("2019-08-01 10:00:02 UTC", "s2", "b", "False"),
        ("2019-08-01 10:00:01 UTC", "s1", "c", "True"),
        ("2019-08-01 09:59:59 UTC", "s1", "a", "False"),
        ("2019-08-01 10:00:00 UTC", "s2", "a", "True"),
        ("2019-08-01 10:00:00 UTC", "s1", "b", "True"),
    ]
    path = tmp_path / "log.csv"
    pd.DataFrame(rows, columns=["timestamp_TW", "uuid", "ucid", "is_correct"]).to_csv(path, index=False)

    inter = load_log(path, DataConfig())

    assert list(inter.student_uuid) == ["s1", "s2"]
    assert list(inter.concept_names) == ["<pad>", "a", "b", "c"]
    concepts, correct = inter.sequence(0)
    assert inter.concept_names[concepts].tolist() == ["a", "b", "c"]
    assert correct.tolist() == [0, 1, 1]
    concepts, correct = inter.sequence(1)
    assert inter.concept_names[concepts].tolist() == ["a", "b"]
    assert correct.tolist() == [1, 0]


def test_attempts_within_a_time_window_are_grouped_by_concept(tmp_path):
    # One student, all in the same 15-minute window; the file order is scrambled.
    rows = [
        ("2019-08-01 10:00:00 UTC", "s1", "b", 2, 1, "True"),
        ("2019-08-01 10:00:00 UTC", "s1", "a", 1, 2, "False"),
        ("2019-08-01 10:00:00 UTC", "s1", "b", 1, 1, "False"),
        ("2019-08-01 10:00:00 UTC", "s1", "a", 1, 1, "True"),
        ("2019-08-01 09:45:00 UTC", "s1", "c", 1, 1, "True"),
    ]
    columns = ["timestamp_TW", "uuid", "ucid", "problem_number", "exercise_problem_repeat_session", "is_correct"]
    path = tmp_path / "log.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)

    inter = load_log(path, DataConfig())

    concepts, correct = inter.sequence(0)
    assert inter.concept_names[concepts].tolist() == ["c", "a", "a", "b", "b"]
    assert correct.tolist() == [1, 1, 0, 0, 1]  # a: session 1 then 2; b: problem 1 then 2


def test_splits_are_disjoint_and_complete(tiny_processed):
    parts = [set(tiny_processed.splits[name].tolist()) for name in SPLITS]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert set().union(*parts) == set(range(tiny_processed.inter.num_students))
    for name in SPLITS:
        assert set(tiny_processed.examples[name].student.tolist()) <= set(tiny_processed.splits[name].tolist())


def test_examples_are_next_new_concepts_with_valid_candidates(tiny_processed):
    cfg = tiny_processed.data_cfg
    for name in SPLITS:
        ex = tiny_processed.examples[name]
        assert len(ex) > 0
        for i in range(len(ex)):
            sequence, _ = tiny_processed.inter.sequence(ex.student[i])
            times = tiny_processed.inter.times(ex.student[i])
            distinct, firsts = first_attempts(sequence)
            cut = ex.cut[i]
            j = np.searchsorted(firsts, cut)  # distinct concepts in the history
            history = sequence[:cut]
            assert cut >= cfg.min_history
            assert times[cut - 1] < times[cut], "the history must end at a time-window boundary"
            assert times[firsts[j]] == times[cut], "the first target starts in the cut's window"
            assert ex.targets[i].tolist() == distinct[j : j + cfg.path_len].tolist()
            assert not np.isin(ex.targets[i], history).any()
            group = ex.first_group[i]
            target_windows = times[firsts[j : j + cfg.path_len]]
            assert (target_windows[:group] == times[cut]).all()
            assert group == cfg.path_len or target_windows[group] != times[cut]
            cands = ex.candidates[i]
            assert len(set(cands.tolist())) == cfg.num_candidates
            assert cands[ex.target_slots[i]].tolist() == ex.targets[i].tolist()
            negatives = np.setdiff1d(cands, ex.targets[i])
            tied = distinct[j + cfg.path_len :][times[firsts[j + cfg.path_len :]] == target_windows[-1]]
            unseen_pool = tiny_processed.inter.num_concepts - (j + cfg.path_len + len(tied))
            if unseen_pool >= cfg.num_candidates - cfg.path_len:  # else the small-vocabulary fallback applies
                assert not np.isin(negatives, history).any(), "exclude_seen must keep seen concepts out"
                assert not np.isin(negatives, tied).any(), "concepts tied with the last target can't be negatives"


def test_popularity_uses_training_students_only(tiny_processed):
    inter, splits = tiny_processed.inter, tiny_processed.splits
    expected = np.zeros(inter.num_concepts + 1)
    for s in splits["train"]:
        expected[np.unique(inter.sequence(s)[0])] += 1
    assert np.array_equal(tiny_processed.popularity, expected)
    assert not np.array_equal(concept_popularity(inter, splits["val"]), expected)


def test_last_window_mode_keeps_each_students_latest_cut(tiny_processed):
    train = tiny_processed.splits["train"]
    pop = tiny_processed.popularity
    last = build_examples(tiny_processed.inter, train, pop, DataConfig(num_candidates=12, windows="last"), seed=0)
    every = build_examples(
        tiny_processed.inter, train, pop, DataConfig(num_candidates=12, max_windows_per_student=10_000), seed=0
    )
    assert len(set(last.student.tolist())) == len(last)
    latest = {}
    for s, cut in zip(every.student.tolist(), every.cut.tolist()):
        latest[s] = max(cut, latest.get(s, 0))
    assert {s: c for s, c in zip(last.student.tolist(), last.cut.tolist())} == latest


def test_popularity_negatives_prefer_popular_concepts(tiny_processed):
    pop = tiny_processed.popularity
    train = tiny_processed.splits["train"]
    kwargs = dict(num_candidates=12, exclude_seen=False)
    uniform = build_examples(tiny_processed.inter, train, pop, DataConfig(**kwargs), seed=0)
    weighted = build_examples(tiny_processed.inter, train, pop, DataConfig(negatives="popularity", **kwargs), seed=0)

    def mean_negative_popularity(ex):
        mask = np.ones(ex.candidates.shape, dtype=bool)
        np.put_along_axis(mask, ex.target_slots, False, axis=1)
        return pop[ex.candidates[mask]].mean()

    assert mean_negative_popularity(weighted) > mean_negative_popularity(uniform)


def test_processed_round_trip(tiny_log, tmp_path):
    cfg = DataConfig(num_candidates=12, max_windows_per_student=4, max_students=120)
    saved = prepare(tiny_log, tmp_path, cfg)
    loaded = load_processed(tmp_path)
    assert loaded.data_cfg == cfg
    assert loaded.stats == saved.stats
    assert loaded.stats["students"] == 120
    for name in SPLITS:
        for field in ("student", "cut", "targets", "candidates", "target_slots"):
            assert np.array_equal(getattr(loaded.examples[name], field), getattr(saved.examples[name], field))
