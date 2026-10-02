"""End-to-end smoke test of the command line on a very small synthetic log."""

import json

from pathtrace.cli import main


def test_synth_prepare_baselines_train_evaluate(tmp_path):
    raw = tmp_path / "raw.csv"
    data = tmp_path / "processed"
    main(["synth", "--out", str(raw), "--students", "80", "--concepts", "40", "--mean-length", "40"])
    main(["prepare", "--log", str(raw), "--out", str(data), "--num-candidates", "10", "--max-windows-per-student", "3"])

    fast = ["--max-epochs", "1", "--bootstrap", "20", "--batch-size", "32", "--max-history", "30"]
    main(["baselines", "--data", str(data), "--out", str(tmp_path / "baselines"), *fast])
    baselines = json.loads((tmp_path / "baselines" / "baselines.json").read_text())
    assert set(baselines["baselines"]) == {"random", "popularity", "markov", "gru_next_item"}

    main(["train", "--data", str(data), "--out", str(tmp_path / "run"), *fast])
    result = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert result["best_epoch"] == 1
    assert {"ci_low", "ci_high"} <= set(result["splits"]["test"]["metrics"]["ndcg"])
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()

    out = tmp_path / "eval.json"
    main(["evaluate", "--data", str(data), "--checkpoint", str(tmp_path / "run" / "model.pt"), "--bootstrap", "20", "--out", str(out)])
    assert json.loads(out.read_text())["n"] == result["splits"]["test"]["n"]
