"""Everything the Colab notebook does, as one resumable command for a local GPU (``learnpath all``).

    download   Junyi's CSVs into <root>/data/raw/junyi (skipped once they are there)
    prepare    the three datasets below (each skipped once prepared)
    grid       each dataset's experiment grid, skipping finished steps and resuming
               an interrupted training run from its last epoch
    report     tables and figures per dataset, into <root>/results/<dataset>
    export     the app bundle from the main dataset's seed-0 pointer network

The main dataset is finished (grid, report, bundle) before the optional ones
start, so its results are usable while the rest runs. Stopping the command
and running it again picks up where it stopped.
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from learnpath.config import DataConfig, ModelConfig, TrainConfig
from learnpath.dkt import DKTConfig

JUNYI_URL = "https://www.kaggle.com/api/v1/datasets/download/junyiacademy/learning-activity-public-dataset-by-junyi-academy"
JUNYI_FILES = ("Log_Problem.csv", "Info_Content.csv", "Info_UserData.csv")

# name -> (prepare overrides, grid options). The first one is the main dataset.
DATASETS: dict[str, tuple[dict, dict]] = {
    "junyi": ({}, {}),
    "junyi-popneg": ({"negatives": "popularity"}, {"variants": ["pointer"], "max_seeds": 3, "sensitivity": False}),
    "junyi-paper": (
        {"windows": "last", "max_students": 2000},
        {"variants": ["pointer"], "max_seeds": 3, "rl": False, "gain": False},
    ),
}


def download_junyi(out_dir: str | Path, url: str = JUNYI_URL, log: Callable[[str], None] = print) -> Path:
    """Download and unzip the Junyi dataset (1.6 GB zip, about 3 GB unpacked); no Kaggle account needed."""
    out = Path(out_dir)
    if all((out / name).exists() for name in JUNYI_FILES):
        log(f"[skip] download: {out} already has {', '.join(JUNYI_FILES)}")
        return out
    out.mkdir(parents=True, exist_ok=True)
    archive, partial = out / "junyi.zip", out / "junyi.zip.part"
    if not archive.exists():
        log(f"[run]  downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "learnpath"})
        with urllib.request.urlopen(request) as response, open(partial, "wb") as f:
            total = int(response.headers.get("Content-Length") or 0)
            copied, step = 0, 200 * 2**20
            while chunk := response.read(2**20):
                f.write(chunk)
                copied += len(chunk)
                if copied % step < len(chunk):
                    log(f"  {copied / 2**30:.1f} GB" + (f" of {total / 2**30:.1f} GB" if total else ""))
        if total and copied != total:
            raise OSError(f"download stopped at {copied:,} of {total:,} bytes; run the command again")
        os.replace(partial, archive)
    log(f"[run]  unzipping {archive.name}")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    missing = [name for name in JUNYI_FILES if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(f"{archive} did not contain {missing}")
    archive.unlink()
    return out


def run_all(
    root: str | Path,
    *,
    datasets: Sequence[str] = tuple(DATASETS),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    dkt_cfg: DKTConfig | None = None,
    data_cfg: DataConfig | None = None,
    gain_max_examples: int | None = 20_000,
    report_bootstrap: int = 1000,
    bundle_dir: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> None:
    from learnpath.export import export_bundle
    from learnpath.grid import run_grid
    from learnpath.preprocess import prepare
    from learnpath.report import build_report

    unknown = set(datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets {sorted(unknown)}; choose from {sorted(DATASETS)}")
    root = Path(root)
    raw = download_junyi(root / "data" / "raw" / "junyi", log=log)
    content = raw / "Info_Content.csv"
    main = next(iter(DATASETS))
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig(batch_size=256)

    for name in datasets:
        overrides, _ = DATASETS[name]
        data = root / "data" / "processed" / name
        if (data / "meta.json").exists():
            log(f"[skip] prepare {name}: already prepared")
        else:
            log(f"[run]  prepare {name}")
            prepare(raw / "Log_Problem.csv", data, dataclasses.replace(data_cfg or DataConfig(), **overrides))

    for name in datasets:
        _, options = DATASETS[name]
        options = dict(options)
        max_seeds = options.pop("max_seeds", None)
        data, runs = root / "data" / "processed" / name, root / "runs" / name
        log(f"===== {name}: experiment grid")
        run_grid(
            data, runs, seeds=list(seeds)[:max_seeds], content=content, model_cfg=model_cfg, train_cfg=train_cfg,
            dkt_cfg=dkt_cfg, gain_max_examples=gain_max_examples, log=log, **options,
        )
        log(f"===== {name}: report")
        build_report(data, runs, root / "results" / name, n_boot=report_bootstrap, log=log)
        if name == main:
            log(f"===== {name}: app bundle")
            export_bundle(
                data, runs, Path(bundle_dir) if bundle_dir else root / "bundle", checkpoint=f"pointer-seed{seeds[0]}",
                names_csv=Path(__file__).resolve().parents[1] / "resources" / "junyi_concepts_en.csv",
                content_csv=content, results_dir=root / "results" / name, log=log,
            )
    log("all finished")
