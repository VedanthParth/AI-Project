"""Run the whole experiment grid for one processed dataset, skipping finished steps.

Layout under ``runs_dir``:

    bkt/  prereq/  dkt/              simulators and structure, fitted on the train split
    baselines/                       random, popularity, markov, gru_next_item
    pointer-seed{s}/                 supervised pointer network
    pointer-nokt-seed{s}/            ablation: no knowledge-tracing head
    pointer-noattn-seed{s}/          ablation: no candidate Transformer
    rl-seed{s}/                      RL fine-tuning of pointer-seed{s}
    rl-sup0-seed0/ rl-sup1-seed0/    sensitivity: supervised weight 0 and 1
    rl-ema-seed0/                    sensitivity: moving-average baseline
    gain_test.json                   DKT learning gain of every method

Each step writes its marker file (``metrics.json``, ``*_summary.json``, ...)
last, so a marker on disk means that step finished. Rerunning the same
command after an interruption (a Colab timeout, say) picks up where it
stopped.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

from pathtrace.config import ModelConfig, TrainConfig
from pathtrace.dkt import DKTConfig
from pathtrace.preprocess import load_processed
from pathtrace.utils import load_json, save_json

VARIANTS: dict[str, dict] = {
    "pointer": {},
    "pointer-nokt": {"kt_weight": 0.0},
    "pointer-noattn": {"attn_layers": 0},
}
SENSITIVITY: dict[str, dict] = {
    "rl-sup0": {"sup_weight": 0.0},
    "rl-sup1": {"sup_weight": 1.0},
    "rl-ema": {"baseline": "ema"},
}
BASELINE_NAMES = ("random", "popularity", "markov", "gru_next_item")


def run_grid(
    data_dir: str | Path,
    runs_dir: str | Path,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    variants: Sequence[str] = tuple(VARIANTS),
    content: str | Path | None = None,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    dkt_cfg: DKTConfig | None = None,
    rl: bool = True,
    sensitivity: bool = True,
    gain: bool = True,
    gain_split: str = "test",
    gain_max_examples: int | None = 20_000,
    log: Callable[[str], None] = print,
) -> None:
    from pathtrace.baselines import run_baselines
    from pathtrace.bkt import BKTParams, fit_and_report
    from pathtrace.dkt import collect_paths, gain_report, load_dkt, train_dkt
    from pathtrace.prereq import PrereqGraph, infer_and_report
    from pathtrace.rl import RLConfig, finetune
    from pathtrace.train import train_model
    from pathtrace.utils import get_device

    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}; choose from {sorted(VARIANTS)}")
    proc = load_processed(data_dir)
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    def step(name: str, marker: Path, action: Callable[[], object]) -> None:
        if marker.exists():
            log(f"[skip] {name}: already done")
            return
        log(f"[run]  {name}")
        action()

    # Simulators and structure.
    step("bkt", runs / "bkt" / "bkt_summary.json", lambda: fit_and_report(proc, runs / "bkt", log=log))
    step(
        "prereq",
        runs / "prereq" / "prereq_summary.json",
        lambda: infer_and_report(proc, runs / "prereq", content_path=content, log=log),
    )
    dkt_cfg = dkt_cfg or DKTConfig(device=train_cfg.device)
    step("dkt", runs / "dkt" / "dkt_summary.json", lambda: train_dkt(proc, dkt_cfg, runs / "dkt", log=log))

    # Baselines, one at a time so an interruption loses at most one.
    baselines_file = runs / "baselines" / "baselines.json"
    done = load_json(baselines_file).get("baselines", {}) if baselines_file.exists() else {}
    for name in BASELINE_NAMES:
        if name in done:
            log(f"[skip] baseline {name}: already done")
            continue
        log(f"[run]  baseline {name}")
        run_baselines(proc, [name], model_cfg, train_cfg, runs / "baselines", log=log)

    # Supervised pointer networks.
    for seed in seeds:
        for variant in variants:
            out = runs / f"{variant}-seed{seed}"
            cfg_m = dataclasses.replace(model_cfg, **VARIANTS[variant])
            cfg_t = dataclasses.replace(train_cfg, seed=seed)
            step(out.name, out / "metrics.json", lambda o=out, m=cfg_m, t=cfg_t: train_model(proc, m, t, o, log=log))

    # RL fine-tuning, then sensitivity runs on seed 0.
    if rl:
        params = BKTParams.load(runs / "bkt" / "bkt_params.npz")
        graph = PrereqGraph.load(runs / "prereq" / "prereq_graph.npz")
        base_rl = RLConfig(
            batch_size=train_cfg.batch_size,
            max_history=train_cfg.max_history,
            bootstrap=train_cfg.bootstrap,
            device=train_cfg.device,
        )
        jobs = [("rl", seed, {}) for seed in seeds]
        if sensitivity:
            jobs += [(name, seeds[0], overrides) for name, overrides in SENSITIVITY.items()]
        for name, seed, overrides in jobs:
            out = runs / f"{name}-seed{seed}"
            checkpoint = runs / f"pointer-seed{seed}" / "model.pt"
            if not checkpoint.exists():
                log(f"[skip] {out.name}: needs {checkpoint}")
                continue
            cfg = dataclasses.replace(base_rl, seed=seed, **overrides)
            step(out.name, out / "metrics.json", lambda o=out, c=cfg, ck=checkpoint: finetune(proc, ck, params, graph, c, o, log=log))

    # DKT learning gain of every method found under runs_dir (recomputed each time).
    if gain:
        log(f"[run]  DKT learning gain on {gain_split}")
        model = load_dkt(runs / "dkt" / "dkt.pt", get_device(train_cfg.device))
        run_dirs = [runs / "baselines"] + sorted(d for d in runs.iterdir() if d.is_dir() and "-seed" in d.name)
        paths = collect_paths(proc, gain_split, run_dirs, runs / "bkt" / "bkt_params.npz")
        report = gain_report(proc, model, gain_split, paths, max_examples=gain_max_examples)
        save_json(report, runs / f"gain_{gain_split}.json")
    log("grid finished")
