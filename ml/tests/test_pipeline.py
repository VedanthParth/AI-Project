import shutil

import pandas as pd

from learnpath.config import DataConfig, ModelConfig, TrainConfig
from learnpath.dkt import DKTConfig
from learnpath.pipeline import JUNYI_FILES, download_junyi, run_all
from learnpath.utils import load_json


def raw_folder(tiny_log, root):
    """A data/raw/junyi folder as the download leaves it, built from the synthetic log."""
    raw = root / "data" / "raw" / "junyi"
    raw.mkdir(parents=True)
    shutil.copyfile(tiny_log, raw / "Log_Problem.csv")
    ucids = sorted(pd.read_csv(tiny_log)["ucid"].astype(str).unique())
    pd.DataFrame(
        {
            "ucid": ucids,
            "content_pretty_name": ucids,
            "learning_stage": ["elementary" if i % 2 else "junior" for i in range(len(ucids))],
            "difficulty": "easy",
            "level2_id": [i % 3 for i in range(len(ucids))],
            "level3_id": [i % 6 for i in range(len(ucids))],
            "level4_id": [i % 12 for i in range(len(ucids))],
        }
    ).to_csv(raw / "Info_Content.csv", index=False)
    (raw / "Info_UserData.csv").write_text("uuid\n", encoding="utf-8")
    return raw


def test_download_is_skipped_when_the_files_are_there(tmp_path):
    for name in JUNYI_FILES:
        (tmp_path / name).write_text("x", encoding="utf-8")
    lines: list[str] = []
    download_junyi(tmp_path, url="http://unreachable.invalid/junyi.zip", log=lines.append)
    assert lines[0].startswith("[skip] download")


def test_run_all_finishes_the_main_dataset_then_resumes_without_redoing_work(tiny_log, tmp_path):
    raw_folder(tiny_log, tmp_path)
    kwargs = dict(
        datasets=["junyi", "junyi-paper"],
        seeds=[0],
        model_cfg=ModelConfig(embed_dim=16, hidden_dim=16, ff_dim=32),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu"),
        dkt_cfg=DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=1, device="cpu"),
        data_cfg=DataConfig(num_candidates=12, max_windows_per_student=4),
        gain_max_examples=50,
        report_bootstrap=20,
    )
    first: list[str] = []
    run_all(tmp_path, log=first.append, **kwargs)

    assert first.index("===== junyi: app bundle") < first.index("===== junyi-paper: experiment grid")
    assert (tmp_path / "results" / "junyi" / "main_table.md").exists()
    assert (tmp_path / "results" / "junyi-paper" / "main_table.md").exists()
    assert load_json(tmp_path / "bundle" / "meta.json")["source"]["checkpoint"] == "rl-seed0"
    assert load_json(tmp_path / "data" / "processed" / "junyi-paper" / "meta.json")["data_config"]["windows"] == "last"
    assert (tmp_path / "runs" / "junyi-paper" / "rl-seed0" / "metrics.json").exists()  # RL is never skipped

    second: list[str] = []
    run_all(tmp_path, log=second.append, **kwargs)
    assert not [line for line in second if line.startswith("[run]  ") and "learning gain" not in line]
