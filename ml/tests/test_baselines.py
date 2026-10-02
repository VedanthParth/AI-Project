import numpy as np
import pandas as pd

from pathtrace import metrics
from pathtrace.baselines import MarkovBaseline, PopularityBaseline, evaluate_baseline
from pathtrace.config import DataConfig
from pathtrace.preprocess import prepare


def test_popularity_scores_come_from_training_counts(tiny_processed):
    scores = PopularityBaseline().score(tiny_processed, "test")
    assert np.array_equal(scores, tiny_processed.popularity[tiny_processed.examples["test"].candidates])


def test_markov_learns_a_fixed_curriculum(tmp_path):
    # Every student walks the same 30-concept chain, so the next concept is
    # always the successor of the last one seen.
    rows = []
    for s in range(40):
        for step in range(30):
            stamp = pd.Timestamp("2019-08-01") + pd.Timedelta(minutes=step)
            rows.append((stamp.strftime("%Y-%m-%d %H:%M:%S UTC"), f"s{s:02d}", f"c{step:02d}", "True"))
    path = tmp_path / "chain.csv"
    pd.DataFrame(rows, columns=["timestamp_TW", "uuid", "ucid", "is_correct"]).to_csv(path, index=False)
    proc = prepare(path, tmp_path / "processed", DataConfig(num_candidates=8, max_windows_per_student=5))

    markov = MarkovBaseline()
    markov.fit(proc)
    result, _ = evaluate_baseline(markov, proc, "test", k=3)
    assert result["metrics"]["first_step"]["mean"] == 1.0
    assert result["metrics"]["precision"]["mean"] > metrics.chance(8, 3)["precision"]
