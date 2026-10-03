"""Command line: ``python -m pathtrace <command>`` (or ``pathtrace <command>``).

    synth      write a synthetic Junyi-shaped log (development and tests only)
    prepare    raw log -> student split + path examples
    baselines  fit and evaluate Random / Popularity / Markov / GRU next-item
    train      supervised training of the pointer network, then val/test evaluation
    bkt        fit per-concept BKT (the RL reward's student simulator)
    prereq     infer a prerequisite graph from first-attempt order
    dkt        train the DKT evaluator (held out from training the recommenders)
    gain       DKT learning gain of recommended paths
    finetune   corrected REINFORCE fine-tuning against the BKT reward
    grid       the full experiment grid for one dataset (resumable)
    report     paper tables and figures from a dataset's runs
    export     bundle a trained model, simulators and sample students for the app
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
        rows = {"chance": results["chance"][split]}
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
            {"chance": results["chance"][split], "pointer": _means(result)},
        )


def cmd_bkt(args: argparse.Namespace) -> None:
    from pathtrace.bkt import fit_and_report
    from pathtrace.preprocess import load_processed

    proc = load_processed(args.data)
    summary = fit_and_report(
        proc, args.out, min_attempts=args.min_attempts, max_guess=args.max_guess, max_slip=args.max_slip
    )
    print("\nheld-out log-likelihood per attempt (val students; higher is better)")
    for name, value in summary["heldout_loglik_per_attempt"].items():
        print(f"  {name:16}{value:10.4f}")
    for split, rows in summary["reward"].items():
        print(f"\nexpected BKT reward of a {proc.data_cfg.path_len}-concept path, {split}")
        print(f"{'':18}{'oracle':>10}{'random':>10}{'target':>10}{'spread':>10}")
        for name, r in rows.items():
            print(
                f"  {name:16}{r['oracle_reward']:10.4f}{r['random_path_reward']:10.4f}"
                f"{r['target_path_reward']:10.4f}{r['gain_spread_within_example']:10.4f}"
            )
    print(f"\nsaved bkt_params.npz and bkt_summary.json to {args.out}")


def cmd_prereq(args: argparse.Namespace) -> None:
    from pathtrace.preprocess import load_processed
    from pathtrace.prereq import infer_and_report

    proc = load_processed(args.data)
    summary = infer_and_report(
        proc,
        args.out,
        content_path=args.content,
        min_support=args.min_support,
        min_precedence=args.min_precedence,
        min_proximity=args.min_proximity,
        proximity_window=args.proximity_window,
    )
    graph = summary["graph"]
    print(
        f"{graph['edges']} edges ({graph['edges_before_reduction']} before transitive reduction); "
        f"{graph['concepts_with_prerequisites']} concepts have prerequisites"
    )
    for name, row in summary.get("agreement_with_content", {}).items():
        print(f"  {name:28} edges {row['edges']:.3f}   random pairs {row['random_pairs']:.3f}")
    for split, row in summary["violations"].items():
        print(
            f"  {split}: violations per path, students {row['student_paths']:.3f} vs random {row['random_paths']:.3f}"
        )
    print(f"saved prereq_graph.npz and prereq_summary.json to {args.out}")


def cmd_dkt(args: argparse.Namespace) -> None:
    from pathtrace.dkt import DKTConfig, train_dkt
    from pathtrace.preprocess import load_processed

    cfg = DKTConfig(hidden_dim=args.hidden_dim, max_epochs=args.max_epochs, seed=args.seed, device=args.device)
    summary = train_dkt(load_processed(args.data), cfg, args.out)
    print(f"\nDKT best epoch {summary['best_epoch']}: val AUC {summary['val']['auc']:.4f}, test AUC {summary['test']['auc']:.4f}")
    print(f"saved dkt.pt and dkt_summary.json to {args.out}")


def cmd_gain(args: argparse.Namespace) -> None:
    from pathtrace.dkt import collect_paths, gain_report, load_dkt
    from pathtrace.preprocess import load_processed
    from pathtrace.utils import get_device

    proc = load_processed(args.data)
    model = load_dkt(args.dkt, get_device(args.device))
    paths = collect_paths(proc, args.split, args.runs, args.bkt)
    report = gain_report(
        proc, model, args.split, paths, max_examples=args.max_examples, attempts=args.attempts, rollouts=args.rollouts
    )
    print(f"\nDKT learning gain E_p on {args.split} ({report['examples']:,} examples)")
    print(f"{'':16}{'before':>10}{'after':>10}{'E_p':>10}{'95% CI':>20}")
    for name, r in sorted(report["methods"].items(), key=lambda kv: -kv[1]["gain"]["mean"]):
        g = r["gain"]
        print(f"{name:16}{r['before']:10.4f}{r['after']:10.4f}{g['mean']:10.4f}   [{g['ci_low']:.4f}, {g['ci_high']:.4f}]")
    if args.out:
        save_json(report, args.out)


def cmd_finetune(args: argparse.Namespace) -> None:
    from pathtrace.bkt import BKTParams
    from pathtrace.preprocess import load_processed
    from pathtrace.prereq import PrereqGraph
    from pathtrace.rl import RLConfig, finetune

    proc = load_processed(args.data)
    results = finetune(
        proc,
        args.checkpoint,
        BKTParams.load(args.bkt),
        PrereqGraph.load(args.prereq),
        dataclass_from_args(RLConfig, args),
        args.out,
    )
    start = results["start"]
    print(f"\nbest epoch {results['best_epoch']} (0 = the supervised checkpoint was not improved)")
    print(f"  val reward at start {start['reward']['mean']:.4f} ({start['share_of_oracle']:.1%} of oracle)")
    for split in ("val", "test"):
        r = results["splits"][split]
        print(
            f"  {split}: reward {r['reward']['mean']:.4f} ({r['share_of_oracle']:.1%} of oracle), "
            f"violations/path {r['violations_per_path']:.3f}, NDCG@3 {r['metrics']['ndcg']['mean']:.4f}"
        )


def cmd_grid(args: argparse.Namespace) -> None:
    from pathtrace.grid import run_grid

    run_grid(
        args.data,
        args.runs,
        seeds=args.seeds,
        variants=args.variants,
        content=args.content,
        model_cfg=dataclass_from_args(ModelConfig, args),
        train_cfg=dataclass_from_args(TrainConfig, args),
        rl=not args.no_rl,
        sensitivity=not args.no_sensitivity,
        gain=not args.no_gain,
        gain_max_examples=args.gain_max_examples,
    )


def cmd_report(args: argparse.Namespace) -> None:
    from pathtrace.report import build_report

    result = build_report(args.data, args.runs, args.out, split=args.split, n_boot=args.bootstrap)
    print((Path(args.out) / "main_table.md").read_text(encoding="utf-8"))
    print(f"strongest baseline: {result['strongest_baseline']}")


def cmd_export(args: argparse.Namespace) -> None:
    from pathtrace.export import export_bundle

    export_bundle(
        args.data,
        args.runs,
        args.out,
        checkpoint=args.checkpoint,
        names_csv=args.names,
        content_csv=args.content,
        split=args.split,
        max_students=args.max_students,
        seed=args.seed,
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
    parser.add_argument(
        "--threads", type=int, default=0,
        help="PyTorch CPU threads (0 = default). Use 1 when running several jobs at once: oversubscribed threads slow everything down sharply",
    )
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

    p = sub.add_parser("bkt", help="fit per-concept BKT on the train split and summarise the reward")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--out", type=Path, required=True, help="output directory (bkt_params.npz, bkt_summary.json)")
    p.add_argument("--min-attempts", type=int, default=50, help="fewer training attempts -> global fallback")
    p.add_argument("--max-guess", type=float, default=0.3)
    p.add_argument("--max-slip", type=float, default=0.3)
    p.set_defaults(func=cmd_bkt)

    p = sub.add_parser("prereq", help="infer a prerequisite graph from training students' first-attempt order")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--out", type=Path, required=True, help="output directory (prereq_graph.npz, prereq_summary.json)")
    p.add_argument("--content", type=Path, default=None, help="Info_Content.csv, to check edges against Junyi's hierarchy")
    p.add_argument("--min-support", type=int, default=30, help="students who started both concepts, in different windows")
    p.add_argument("--min-precedence", type=float, default=0.9, help="share of them who started the prerequisite first")
    p.add_argument("--min-proximity", type=float, default=0.2, help="share who started the dependent concept soon after")
    p.add_argument("--proximity-window", type=int, default=5, help="'soon' = within this many new concepts")
    p.set_defaults(func=cmd_prereq)

    p = sub.add_parser("dkt", help="train the DKT evaluator on the train split")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--out", type=Path, required=True, help="output directory (dkt.pt, dkt_summary.json)")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_dkt)

    p = sub.add_parser("gain", help="DKT learning gain of saved paths (plus random, student and BKT-oracle paths)")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--dkt", type=Path, required=True, help="dkt.pt from the dkt command")
    p.add_argument("--runs", type=Path, nargs="*", default=[], help="run directories holding preds*_<split>.npy")
    p.add_argument("--bkt", type=Path, default=None, help="bkt_params.npz, to include the BKT oracle's paths")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--max-examples", type=int, default=None, help="score a fixed random subset of examples")
    p.add_argument("--attempts", type=int, default=3, help="simulated attempts per path concept")
    p.add_argument("--rollouts", type=int, default=8, help="simulations averaged per example")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    p.set_defaults(func=cmd_gain)

    from pathtrace.rl import RLConfig

    p = sub.add_parser("finetune", help="REINFORCE fine-tuning of a trained pointer network against the BKT reward")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--checkpoint", type=Path, required=True, help="supervised model.pt to start from")
    p.add_argument("--bkt", type=Path, required=True, help="bkt_params.npz from the bkt command")
    p.add_argument("--prereq", type=Path, required=True, help="prereq_graph.npz from the prereq command")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    add_dataclass_args(p, RLConfig)
    p.set_defaults(func=cmd_finetune)

    from pathtrace.grid import VARIANTS

    p = sub.add_parser("grid", help="run the full experiment grid for one dataset, skipping finished steps")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--runs", type=Path, required=True, help="root directory for all runs of this dataset")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--content", type=Path, default=None, help="Info_Content.csv for the prerequisite-graph checks")
    p.add_argument("--no-rl", action="store_true", help="skip RL fine-tuning")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the seed-0 RL sensitivity runs")
    p.add_argument("--no-gain", action="store_true", help="skip the DKT learning-gain evaluation")
    p.add_argument("--gain-max-examples", type=int, default=20_000, help="test examples scored for learning gain")
    add_dataclass_args(p, ModelConfig)
    add_dataclass_args(p, TrainConfig)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("report", help="paper tables (Markdown, CSV, LaTeX) and figures from a dataset's runs")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--runs", type=Path, required=True, help="the runs directory the grid wrote")
    p.add_argument("--out", type=Path, required=True, help="output directory for tables and figures")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="bundle a trained model, its simulators and sample students for the app")
    p.add_argument("--data", type=Path, required=True, help="processed data directory")
    p.add_argument("--runs", type=Path, required=True, help="runs directory with bkt/, prereq/, dkt/ and the checkpoint")
    p.add_argument("--out", type=Path, required=True, help="bundle directory (replaced if it exists)")
    p.add_argument("--checkpoint", default="pointer-seed0", help="run directory under --runs holding model.pt")
    p.add_argument("--names", type=Path, default=Path(__file__).resolve().parents[1] / "resources" / "junyi_concepts_en.csv",
                   help="CSV with ucid, name_en and level columns")
    p.add_argument("--content", type=Path, default=None, help="Junyi Info_Content.csv, for learning stage and difficulty")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--max-students", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("evaluate", help="evaluate a saved checkpoint")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=None, help="optional JSON output path")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
