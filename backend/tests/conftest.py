import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A small app bundle exported from synthetic data, the same way the real one is."""
    from pathtrace.config import DataConfig, ModelConfig, TrainConfig
    from pathtrace.dkt import DKTConfig
    from pathtrace.export import export_bundle
    from pathtrace.grid import run_grid
    from pathtrace.preprocess import prepare
    from pathtrace.synthetic import write_log

    root = tmp_path_factory.mktemp("app")
    log = write_log(root / "Log_Problem.csv", num_students=150, num_concepts=40, mean_length=40, seed=5)
    prepare(log, root / "data", DataConfig(num_candidates=10, max_windows_per_student=3))
    run_grid(
        root / "data",
        root / "runs",
        seeds=[0],
        variants=["pointer"],
        model_cfg=ModelConfig(embed_dim=8, hidden_dim=8, ff_dim=16),
        train_cfg=TrainConfig(max_epochs=1, bootstrap=0, max_history=20, batch_size=32),
        dkt_cfg=DKTConfig(hidden_dim=8, embed_dim=8, max_epochs=1),
        rl=False,
        gain=False,
        log=lambda _: None,
    )
    names = Path(__file__).resolve().parents[2] / "ml" / "resources" / "junyi_concepts_en.csv"
    export_bundle(root / "data", root / "runs", root / "bundle", names_csv=names, max_students=8, log=lambda _: None)
    return root / "bundle"
