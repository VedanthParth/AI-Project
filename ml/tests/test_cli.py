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
    baselines = json.loads((tmp_path / "baselines" / "baselines.json").read_text(encoding="utf-8"))
    assert set(baselines["baselines"]) == {"random", "popularity", "markov", "gru_next_item"}

    main(["train", "--data", str(data), "--out", str(tmp_path / "run"), *fast])
    result = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert result["best_epoch"] == 1
    assert {"ci_low", "ci_high"} <= set(result["splits"]["test"]["metrics"]["ndcg"])
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()

    main(["bkt", "--data", str(data), "--out", str(tmp_path / "bkt"), "--min-attempts", "10"])
    main(["prereq", "--data", str(data), "--out", str(tmp_path / "prereq"), "--min-support", "5"])
    assert (tmp_path / "prereq" / "prereq_graph.npz").exists()
    main(["dkt", "--data", str(data), "--out", str(tmp_path / "dkt"), "--max-epochs", "1", "--hidden-dim", "16"])
    gain = tmp_path / "gain.json"
    main([
        "gain", "--data", str(data), "--dkt", str(tmp_path / "dkt" / "dkt.pt"), "--runs", str(tmp_path / "baselines"),
        str(tmp_path / "run"), "--bkt", str(tmp_path / "bkt" / "bkt_params.npz"), "--rollouts", "2", "--out", str(gain),
    ])
    methods = json.loads(gain.read_text(encoding="utf-8"))["methods"]
    assert {"random", "students", "popularity", "run", "bkt_oracle"} <= set(methods)
    main([
        "finetune", "--data", str(data), "--checkpoint", str(tmp_path / "run" / "model.pt"),
        "--bkt", str(tmp_path / "bkt" / "bkt_params.npz"), "--prereq", str(tmp_path / "prereq" / "prereq_graph.npz"),
        "--out", str(tmp_path / "rl"), "--max-epochs", "1", "--bootstrap", "20", "--max-history", "30",
    ])
    assert json.loads((tmp_path / "rl" / "metrics.json").read_text(encoding="utf-8"))["splits"]["test"]["n"] > 0
    summary = json.loads((tmp_path / "bkt" / "bkt_summary.json").read_text(encoding="utf-8"))
    assert summary["fit"]["concepts_fitted"] > 0
    assert (tmp_path / "bkt" / "bkt_params.npz").exists()

    out = tmp_path / "eval.json"
    main(["evaluate", "--data", str(data), "--checkpoint", str(tmp_path / "run" / "model.pt"), "--bootstrap", "20", "--out", str(out)])
    assert json.loads(out.read_text(encoding="utf-8"))["n"] == result["splits"]["test"]["n"]
