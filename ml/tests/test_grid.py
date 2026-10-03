from pathtrace.config import DataConfig, ModelConfig, TrainConfig
from pathtrace.dkt import DKTConfig
from pathtrace.grid import run_grid
from pathtrace.preprocess import prepare
from pathtrace.utils import load_json


def test_grid_runs_everything_then_resumes_without_redoing_work(tiny_log, tmp_path):
    data = tmp_path / "processed"
    prepare(tiny_log, data, DataConfig(num_candidates=12, max_windows_per_student=4))
    kwargs = dict(
        seeds=[0],
        variants=["pointer", "pointer-nokt"],
        model_cfg=ModelConfig(embed_dim=16, hidden_dim=16, ff_dim=32),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32),
        dkt_cfg=DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=1),
        gain_max_examples=50,
    )
    first: list[str] = []
    run_grid(data, tmp_path / "runs", log=first.append, **kwargs)

    runs = tmp_path / "runs"
    for marker in [
        "bkt/bkt_summary.json",
        "prereq/prereq_summary.json",
        "dkt/dkt_summary.json",
        "pointer-seed0/metrics.json",
        "pointer-nokt-seed0/metrics.json",
        "rl-seed0/metrics.json",
        "rl-ema-seed0/metrics.json",
    ]:
        assert (runs / marker).exists(), marker
    assert set(load_json(runs / "baselines" / "baselines.json")["baselines"]) == {
        "random", "popularity", "markov", "gru_next_item"
    }
    gain_methods = load_json(runs / "gain_test.json")["methods"]
    assert {"pointer-seed0", "rl-seed0", "markov", "bkt_oracle"} <= set(gain_methods)

    from pathtrace.report import build_report

    result = build_report(data, runs, tmp_path / "results", n_boot=50, log=lambda _: None)
    assert {"random", "markov", "pointer", "pointer-nokt", "rl", "rl-ema", "students", "bkt_oracle"} <= set(result["rows"])
    assert result["rows"]["bkt_oracle"]["reward_share"][0] == 1.0
    assert "vs_strongest" in result["rows"]["pointer"]
    for name in ("main_table.md", "main_table.csv", "main_table.tex", "results.json", "fig_training.png", "fig_rl.png", "fig_methods.png"):
        assert (tmp_path / "results" / name).exists(), name

    second: list[str] = []
    run_grid(data, tmp_path / "runs", log=second.append, **kwargs)
    assert not [line for line in second if line.startswith("[run]  ") and "learning gain" not in line]
