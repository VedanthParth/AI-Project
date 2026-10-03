"""Turn a dataset's run folders into the paper's tables and figures.

Reads what ``grid`` writes under ``runs_dir`` and produces, in ``out_dir``:

- ``main_table.md`` / ``.csv`` / ``.tex``: one row per method, mean +- std
  over seeds, with ranking metrics, BKT reward (share of the oracle),
  prerequisite violations, DKT learning gain, and the NDCG@3 difference to
  the strongest baseline with a paired bootstrap 95% CI and p-value;
- ``results.json``: everything in the table, unrounded;
- ``fig_training.png``, ``fig_rl.png``, ``fig_methods.png`` when the inputs
  exist.

Every number comes from saved predictions, re-scored here, so the table can
always be regenerated from the run folders.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pathtrace import metrics
from pathtrace.preprocess import Processed, load_processed
from pathtrace.utils import load_json, save_json

BASELINES = ("random", "popularity", "markov", "gru_next_item")
LABELS = {
    "random": "Random",
    "popularity": "Popularity",
    "markov": "Markov",
    "gru_next_item": "GRU next-item",
    "pointer": "Pointer network",
    "pointer-nokt": "  without KT head",
    "pointer-noattn": "  without candidate Transformer",
    "rl": "Pointer network + RL",
    "rl-sup0": "  RL, supervised weight 0",
    "rl-sup1": "  RL, supervised weight 1",
    "rl-ema": "  RL, moving-average baseline",
    "students": "Students' own next concepts",
    "bkt_oracle": "BKT oracle",
}
ORDER = list(LABELS)
METRIC_COLUMNS = [("precision", "P@3"), ("hit_rate", "HR@3"), ("ndcg", "NDCG@3"), ("mrr", "MRR"), ("first_step", "First step")]

# Validated light palette (dataviz reference instance); ours vs baselines by emphasis.
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def _collect_predictions(runs: Path, split: str) -> dict[str, list[np.ndarray]]:
    found: dict[str, list[np.ndarray]] = {}
    for f in sorted((runs / "baselines").glob(f"preds_*_{split}.npy")):
        found[f.stem[len("preds_") : -len(f"_{split}")]] = [np.load(f)]
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        match = re.fullmatch(r"(.+)-seed(\d+)", d.name)
        if match and (d / f"preds_{split}.npy").exists():
            found.setdefault(match[1], []).append(np.load(d / f"preds_{split}.npy"))
    return found


def _gain_by_method(runs: Path, split: str) -> dict[str, list[float]]:
    path = runs / f"gain_{split}.json"
    if not path.exists():
        return {}
    grouped: dict[str, list[float]] = {}
    for name, row in load_json(path)["methods"].items():
        match = re.fullmatch(r"(.+)-seed(\d+)", name)
        grouped.setdefault(match[1] if match else name, []).append(row["gain"]["mean"])
    return grouped


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else None)


def _paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": float(min(1.0, p))}


def build_report(
    data_dir: str | Path,
    runs_dir: str | Path,
    out_dir: str | Path,
    split: str = "test",
    n_boot: int = 1000,
    violation_weight: float = 0.2,
    figures: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    proc = load_processed(data_dir)
    runs, out = Path(runs_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = proc.examples[split]
    k = proc.data_cfg.path_len
    idx = np.arange(len(ex))

    preds = _collect_predictions(runs, split)
    tables = oracle = None
    if (runs / "bkt" / "bkt_params.npz").exists() and (runs / "prereq" / "prereq_graph.npz").exists():
        from pathtrace import bkt, prereq
        from pathtrace.rl import build_reward_tables, oracle_paths

        tables = build_reward_tables(
            proc,
            split,
            bkt.BKTParams.load(runs / "bkt" / "bkt_params.npz"),
            prereq.PrereqGraph.load(runs / "prereq" / "prereq_graph.npz"),
            violation_weight,
        )
        oracle = oracle_paths(tables, k)
        preds["bkt_oracle"] = [oracle]
        preds["students"] = [ex.target_slots]
        oracle_mean = float(tables.reward(idx, oracle).mean())
    gains = _gain_by_method(runs, split)

    per_example_ndcg: dict[str, np.ndarray] = {}
    rows: dict[str, dict] = {}
    for name, seed_preds in preds.items():
        row: dict = {"label": LABELS.get(name, name), "seeds": len(seed_preds)}
        per_seed = [metrics.per_example(p, ex.target_slots, ex.first_group) for p in seed_preds]
        if name != "students":
            for key, _ in METRIC_COLUMNS:
                row[key] = _mean_std([v[key].mean() for v in per_seed])
            per_example_ndcg[name] = np.mean([v["ndcg"] for v in per_seed], axis=0)
        if tables is not None:
            rewards = [tables.reward(idx, p).mean() for p in seed_preds]
            row["reward_share"] = _mean_std([r / oracle_mean for r in rewards])
            row["violations"] = _mean_std([tables.violations(idx, p).mean() for p in seed_preds])
        if name in gains:
            row["dkt_gain"] = _mean_std(gains[name])
        rows[name] = row

    present_baselines = [b for b in BASELINES if b in per_example_ndcg and b != "random"]
    strongest = max(present_baselines, key=lambda b: per_example_ndcg[b].mean()) if present_baselines else None
    for name, ndcg in per_example_ndcg.items():
        rows[name]["ndcg_ci"] = metrics.summarize({"ndcg": ndcg}, n_boot)["ndcg"]
        if strongest is not None and name != strongest and name not in BASELINES:
            rows[name]["vs_strongest"] = _paired_bootstrap(ndcg - per_example_ndcg[strongest], n_boot)

    ordered = sorted(rows, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
    chance = metrics.split_chance(proc.data_cfg.num_candidates, k, ex.first_group)
    result = {
        "split": split,
        "examples": len(ex),
        "strongest_baseline": strongest,
        "oracle_reward": oracle_mean if tables is not None else None,
        "chance": chance,
        "rows": {name: rows[name] for name in ordered},
    }
    save_json(result, out / "results.json")
    _write_tables(result, out)
    log(f"wrote main_table.md/.csv/.tex and results.json to {out}")
    if figures:
        _figures(proc, runs, result, per_example_ndcg, out, log)
    return result


# --------------------------------------------------------------------------- tables


def _fmt(value, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "—"
    mean, std = value
    scale = 100 if percent else 1
    text = f"{mean * scale:.{1 if percent else digits}f}"
    if std is not None:
        text += f" ± {std * scale:.{1 if percent else digits}f}"
    return text + ("%" if percent else "")


def _table_rows(result: dict) -> tuple[list[str], list[list[str]]]:
    header = ["Method", "Seeds"] + [label for _, label in METRIC_COLUMNS]
    header += ["BKT reward (% of oracle)", "Violations / path", "DKT gain E_p", "ΔNDCG@3 vs strongest baseline"]
    chance = result["chance"]
    body = [["Chance (analytic)", "—"] + [f"{chance[key]:.3f}" for key, _ in METRIC_COLUMNS] + ["—"] * 4]
    for row in result["rows"].values():
        vs = row.get("vs_strongest")
        if vs is None:
            vs_text = "—"
        else:
            p_text = "p<0.001" if vs["p_value"] < 0.001 else f"p={vs['p_value']:.3f}"
            vs_text = f"{vs['mean']:+.3f} [{vs['ci_low']:+.3f}, {vs['ci_high']:+.3f}], {p_text}"
        label = "– " + row["label"].strip() if row["label"].startswith("  ") else row["label"]
        body.append(
            [label, str(row["seeds"])]
            + [_fmt(row.get(key)) for key, _ in METRIC_COLUMNS]
            + [_fmt(row.get("reward_share"), percent=True), _fmt(row.get("violations"), 2), _fmt(row.get("dkt_gain"), 4), vs_text]
        )
    return header, body


def _write_tables(result: dict, out: Path) -> None:
    header, body = _table_rows(result)
    caption = (
        f"{result['split'].capitalize()} split, {result['examples']:,} examples. Mean ± std over seeds. "
        f"Strongest baseline: {LABELS.get(result['strongest_baseline'], result['strongest_baseline'])}."
    )
    md = [f"| {' | '.join(header)} |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md += [f"| {' | '.join(r)} |" for r in body]
    (out / "main_table.md").write_text(caption + "\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    with open(out / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)

    def tex(cell: str) -> str:
        return (
            cell.replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("_", "\\_").replace("±", "$\\pm$")
            .replace("Δ", "$\\Delta$").replace("–", "--").replace("—", "--")
        )

    lines = [
        "\\begin{table*}[t]",
        "\\centering\\small",
        f"\\caption{{{tex(caption)}}}",
        "\\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(tex(h) for h in header) + " \\\\",
        "\\midrule",
        *[" & ".join(tex(c) for c in r) + " \\\\" for r in body],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ]
    (out / "main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- figures


def _style(ax, integer_x: bool = False) -> None:
    from matplotlib.ticker import MaxNLocator

    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _seed_histories(runs: Path, variant: str) -> list[list[dict]]:
    return [
        load_json(d / "history.json")
        for d in sorted(runs.glob(f"{variant}-seed*"))
        if re.fullmatch(rf"{re.escape(variant)}-seed\d+", d.name) and (d / "history.json").exists()
    ]


def _band(ax, histories: list[list[dict]], key: str, color: str, label: str) -> None:
    length = min(len(h) for h in histories)
    epochs = [r["epoch"] for r in histories[0][:length]]
    values = np.array([[r[key] for r in h[:length]] for h in histories])
    ax.plot(epochs, values.mean(axis=0), color=color, linewidth=2, label=label)
    if len(histories) > 1:
        ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)


def _figures(proc: Processed, runs: Path, result: dict, ndcg: dict[str, np.ndarray], out: Path, log) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed; skipping figures (pip install matplotlib)")
        return
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white"})

    histories = _seed_histories(runs, "pointer")
    if histories:
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
        _band(left, histories, "train_path_loss", SERIES_1, "Training")
        _band(left, histories, "val_path_loss", SERIES_2, "Validation")
        left.set(title="Path loss (cross-entropy)", xlabel="Epoch")
        left.legend(frameon=False, fontsize=9, labelcolor=INK_2)
        _band(right, histories, "val_ndcg", SERIES_1, "Pointer network")
        baselines_file = runs / "baselines" / "baselines.json"
        baselines = load_json(baselines_file)["baselines"] if baselines_file.exists() else {}
        for name in ("markov", "gru_next_item"):
            if name in baselines:
                value = baselines[name]["val"]["metrics"]["ndcg"]["mean"]
                right.axhline(value, color=MUTED, linewidth=1, linestyle="--")
                right.annotate(LABELS[name], (1, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                               textcoords="offset points", ha="right", fontsize=8, color=INK_2)
        right.set(title="Validation NDCG@3", xlabel="Epoch")
        for ax in (left, right):
            _style(ax, integer_x=True)
        seeds = f"mean of {len(histories)} seeds, band = min to max" if len(histories) > 1 else "one seed"
        fig.suptitle(f"Supervised training ({seeds})", fontsize=10, color=INK_2, y=1.02)
        fig.savefig(out / "fig_training.png")
        plt.close(fig)

    rl_runs = [d for d in sorted(runs.glob("rl-seed*")) if (d / "history.json").exists() and (d / "metrics.json").exists()]
    if rl_runs:
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4))
        shares, ndcgs = [], []
        for d in rl_runs:
            history, start = load_json(d / "history.json"), load_json(d / "metrics.json")["start"]
            shares.append([start["share_of_oracle"]] + [r["val_share_of_oracle"] for r in history[1:]])
            ndcgs.append([r["val_ndcg"] for r in history])
        for ax, series, title in ((left, shares, "Validation BKT reward (share of oracle)"), (right, ndcgs, "Validation NDCG@3")):
            length = min(len(s) for s in series)
            values = np.array([s[:length] for s in series])
            epochs = np.arange(length)
            ax.plot(epochs, values.mean(axis=0), color=SERIES_1, linewidth=2, marker="o", markersize=4)
            if len(series) > 1:
                ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=SERIES_1, alpha=0.15, linewidth=0)
            ax.set(title=title, xlabel="RL epoch (0 = supervised start)")
            _style(ax, integer_x=True)
        left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        seeds = f"mean of {len(rl_runs)} seeds, band = min to max" if len(rl_runs) > 1 else "one seed"
        fig.suptitle(f"RL fine-tuning ({seeds})", fontsize=10, color=INK_2, y=1.02)
        fig.savefig(out / "fig_rl.png")
        plt.close(fig)

    names = [n for n in result["rows"] if n in ndcg and n != "random"]
    if names:
        has_gain = any("dkt_gain" in result["rows"][n] for n in names)
        fig, axes = plt.subplots(1, 2 if has_gain else 1, figsize=(9 if has_gain else 5, 0.42 * len(names) + 1.2), sharey=True)
        axes = np.atleast_1d(axes)
        ypos = np.arange(len(names))[::-1]
        for y, name in zip(ypos, names):
            row = result["rows"][name]
            color = MUTED if name in BASELINES or name in ("students", "bkt_oracle") else SERIES_1
            ci = row.get("ndcg_ci", {})
            if ci:
                axes[0].errorbar(ci["mean"], y, xerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]],
                                 fmt="o", color=color, markersize=6, capsize=0, linewidth=1.5)
            if has_gain and "dkt_gain" in row:
                axes[1].plot(row["dkt_gain"][0], y, "o", color=color, markersize=6)
        axes[0].set_yticks(ypos, [result["rows"][n]["label"].strip() for n in names])
        axes[0].set(title="NDCG@3 (95% CI)")
        if has_gain:
            axes[1].set(title="DKT learning gain E_p")
            axes[1].axvline(0, color=AXIS, linewidth=1)
        for ax in axes:
            _style(ax)
        fig.suptitle(
            "Blue: pointer-network variants. Grey: baselines and references.", fontsize=9, color=INK_2, y=1.0
        )
        fig.savefig(out / "fig_methods.png")
        plt.close(fig)
    log(f"wrote figures to {out}")
