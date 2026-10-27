import numpy as np
import pytest

from learnpath.config import DataConfig, ModelConfig, TrainConfig
from learnpath.dkt import DKTConfig
from learnpath.grid import run_grid
from learnpath.preprocess import concept_popularity, load_log, load_processed
from learnpath.temporal import DAY, fold_report, new_students, prepare_fold, seconds

ORIGIN = "2020-03-01"  # the synthetic log runs from August 2019 to July 2020


@pytest.fixture(scope="module")
def fold(tiny_log, tmp_path_factory):
    cfg = DataConfig(num_candidates=12, max_windows_per_student=4)
    out = tmp_path_factory.mktemp("fold")
    prepare_fold(load_log(tiny_log, cfg), out, cfg, ORIGIN, val_days=30, test_days=90)
    return load_processed(out), load_log(tiny_log, cfg), out


def test_each_split_only_holds_attempts_from_its_period(fold):
    proc, _, _ = fold
    t0 = seconds(ORIGIN)
    ends = {"train": t0 - 30 * DAY, "val": t0, "test": t0 + 90 * DAY}
    for split, end in ends.items():
        for s in proc.splits[split]:
            assert proc.inter.times(s).max() < end
    assert proc.stats["origin"] == ORIGIN and len(proc.examples["test"]) > 0


def test_targets_fall_in_their_window(fold):
    proc, _, _ = fold
    t0 = seconds(ORIGIN)
    lower = {"val": t0 - 30 * DAY, "test": t0}
    for split, low in lower.items():
        ex = proc.examples[split]
        for s, cut in zip(ex.student, ex.cut):
            # the first target's window starts where the history ends
            assert proc.inter.times(s)[cut] >= low


def test_training_statistics_see_only_the_training_period(fold):
    proc, full, _ = fold
    cutoff = seconds(ORIGIN) - 30 * DAY
    before = full.time < cutoff
    owner = np.repeat(np.arange(full.num_students), np.diff(full.offsets))
    expected = np.zeros(full.num_concepts + 1)
    pairs = np.unique(owner[before].astype(np.int64) * (full.num_concepts + 1) + full.concept[before])
    np.add.at(expected, pairs % (full.num_concepts + 1), 1)
    assert np.array_equal(proc.popularity, expected)
    assert np.array_equal(proc.popularity, concept_popularity(proc.inter, proc.splits["train"]))


def test_grid_runs_on_a_fold_and_reports_new_and_continuing_students(fold, tmp_path):
    proc, _, data_dir = fold
    run_grid(
        data_dir, tmp_path / "runs", seeds=[0], variants=["pointer"], sensitivity=False, gain=False,
        model_cfg=ModelConfig(embed_dim=16, hidden_dim=16, ff_dim=32),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu"),
        dkt_cfg=DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=1, device="cpu"), log=lambda _: None,
    )
    report = fold_report(proc, tmp_path / "runs", n_boot=20)
    assert {"markov", "popularity", "pointer"} <= set(report["methods"])
    assert report["examples"]["new"] + report["examples"]["continuing"] == report["examples"]["all"]
    assert report["examples"]["new"] == int(new_students(proc).sum())
