"""Command line: ``python -m pathtrace <command>`` (or ``pathtrace <command>``).

    synth      write a synthetic Junyi-shaped log (development and tests only)
    prepare    raw log -> student split + path examples
    baselines  fit and evaluate Random / Popularity / Markov / GRU next-item
    train      supervised training of the pointer network, then val/test evaluation
    evaluate   re-evaluate a saved checkpoint on one split
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pathtrace import metrics
from pathtrace.config import DataConfig, ModelConfig, TrainConfig, add_dataclass_args, dataclass_from_args
from pathtrace.utils import save_json


def _print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"{'':16}" + "".join(f"{name:>12}" for name in metrics.METRICS))
    for label, values in rows.items():
        print(f"{label:16}" + "".join(f"{values[name]:12.4f}" for name in metrics.METRICS))


def _means(result: dict) -> dict[str, float]:
    return {name: entry["mean"] for name, entry in result["metrics"].items()}


def cmd_synth(args: argparse.Namespace) -> None:
    from pathtrace.synthetic import write_log

    path = write_log(
        args.out, num_students=args.students, num_concepts=args.concepts, mean_length=args.mean_length, seed=args.seed
    )
    print(f"wrote {path}")


def cmd_prepare(args: argparse.Namespace) -> None:
    from pathtrace.preprocess import prepare

    proc = prepare(args.log, args.out, dataclass_from_args(DataConfig, args))
    for key, value in proc.stats.items():
        print(f"{key:>20}: {value:,}" if isinstance(value, int) else f"{key:>20}: {value:.4f}")
    print(f"saved to {args.out}")


def cmd_baselines(args: argparse.Namespace) -> None:
    from pathtrace.baselines import BASELINES, run_baselines
    from pathtrace.preprocess import load_processed

    proc = load_processed(args.data)
    names = args.only.split(",") if args.only else list(BASELINES)
    results = run_baselines(
        proc, names, dataclass_from_args(ModelConfig, args), dataclass_from_args(TrainConfig, args), args.out
    )
    for split in ("val", "test"):
        rows = {"chance": results["chance"]}
        rows.update({name: _means(r[split]) for name, r in results["baselines"].items()})
        _print_table(f"{split} ({proc.stats[f'{split}_examples']:,} examples)", rows)


def cmd_train(args: argparse.Namespace) -> None:
    from pathtrace.preprocess import load_processed
    from pathtrace.train import train_model

    proc = load_processed(args.data)
    results = train_model(proc, dataclass_from_args(ModelConfig, args), dataclass_from_args(TrainConfig, args), args.out)
    print(f"\nbest epoch {results['best_epoch']}, {results['parameters']:,} parameters")
    for split in ("val", "test"):
        result = results["splits"][split]
        _print_table(
            f"{split} ({result['n']:,} examples, KT AUC {result['kt_auc']:.4f})",
            {"chance": results["chance"], "pointer": _means(result)},
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from pathtrace.config import dataclass_from_dict
    from pathtrace.preprocess import load_processed
    from pathtrace.train import evaluate_model, load_checkpoint
    from pathtrace.utils import get_device

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = dataclass_from_dict(TrainConfig, {**ckpt["train_config"], "bootstrap": args.bootstrap})
    proc = load_processed(args.data)
    result, _ = evaluate_model(model, proc, args.split, cfg, device, n_boot=args.bootstrap)
    _print_table(f"{args.split} ({result['n']:,} examples, KT AUC {result['kt_auc']:.4f})", {"pointer": _means(result)})
    if args.out:
        save_json(result, args.out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pathtrace", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic Junyi-shaped log")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--students", type=int, default=1500)
    p.add_argument("--concepts", type=int, default=120)
    p.add_argument("--mean-length", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("prepare", help="raw log -> student split + path examples")
    p.add_argument("--log", type=Path, required=True, help="path to Log_Problem.csv (or a compatible CSV)")
    p.add_argument("--out", type=Path, required=True, help="output directory for processed data")
    add_dataclass_args(p, DataConfig)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("baselines", help="fit and evaluate the baselines")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--out", type=Path, required=True, help="output directory for results")
    p.add_argument("--only", default="", help="comma-separated subset of baselines")
    add_dataclass_args(p, ModelConfig)
    add_dataclass_args(p, TrainConfig)
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("train", help="train the pointer network")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--out", type=Path, required=True, help="run directory (checkpoint, history, metrics)")
    add_dataclass_args(p, ModelConfig)
    add_dataclass_args(p, TrainConfig)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a saved checkpoint")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
