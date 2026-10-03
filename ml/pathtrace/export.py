"""Export what the demo app needs into one small directory (the app bundle).

    bundle/
      data/              a processed dataset holding only the sampled students
                         (``load_processed`` reads it; train and val are empty)
      model.pt           pointer-network checkpoint
      dkt.pt             DKT simulator
      bkt_params.npz     per-concept BKT parameters
      prereq_graph.npz   inferred prerequisite graph
      markov.npz         first-attempt transition counts over training students
      concepts.json      display metadata, one entry per concept id
      meta.json          where everything came from, and its test metrics
      README.md          provenance and license

Every model and table is fitted on the source dataset's train split, and the
sampled students come from its test split, so the app only shows students that
no model was trained on.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pathtrace.baselines import transition_counts
from pathtrace.preprocess import SPLITS, Examples, Interactions, Processed, load_processed, save_processed
from pathtrace.utils import load_json, save_json

DATASET = {
    "name": "Junyi Academy Online Learning Activity Dataset",
    "url": "https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy",
    "license": "CC BY-NC-SA 4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}

README = """# PathTrace app bundle

Exported by `pathtrace export` from `{data}` and `{runs}` on {date}. See `meta.json` for
the exact runs and their test metrics.

The student histories in `data/` are {students} test-split students from the
[{name}]({url}), which is licensed [{license}]({license_url}). This bundle is derived
from it and shared under the same license. Concept names are machine-translated
(see `ml/resources/README.md`).
"""


def _empty_examples(k: int, n: int) -> Examples:
    return Examples(
        student=np.zeros(0, np.int32),
        cut=np.zeros(0, np.int32),
        targets=np.zeros((0, k), np.int32),
        candidates=np.zeros((0, n), np.int32),
        target_slots=np.zeros((0, k), np.int64),
        first_group=np.zeros(0, np.int8),
    )


def subset_students(proc: Processed, split: str, students: np.ndarray) -> Processed:
    """A processed dataset with only ``students`` (renumbered 0..) and their ``split`` examples."""
    inter = proc.inter
    lengths = inter.offsets[students + 1] - inter.offsets[students]
    rows = np.concatenate([np.arange(inter.offsets[s], inter.offsets[s + 1]) for s in students])
    sub_inter = Interactions(
        student_uuid=np.array([f"student-{i + 1:03d}" for i in range(len(students))]),
        concept_names=inter.concept_names,
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        concept=inter.concept[rows],
        correct=inter.correct[rows],
        time=inter.time[rows],
    )
    ex = proc.examples[split]
    new_id = np.full(inter.num_students, -1, dtype=np.int64)
    new_id[students] = np.arange(len(students))
    keep = new_id[ex.student] >= 0
    fields = {f.name: getattr(ex, f.name)[keep] for f in dataclasses.fields(Examples)}
    fields["student"] = new_id[ex.student[keep]].astype(np.int32)
    k, n = ex.targets.shape[1], ex.candidates.shape[1]
    examples = {name: _empty_examples(k, n) for name in SPLITS}
    examples[split] = Examples(**fields)
    splits = {name: np.zeros(0, np.int64) for name in SPLITS}
    splits[split] = np.arange(len(students), dtype=np.int64)
    stats = {"students": len(students), "interactions": int(len(rows)), "examples": int(keep.sum())}
    return Processed(sub_inter, splits, examples, proc.popularity, proc.data_cfg, stats)


def concept_table(concept_names: np.ndarray, popularity: np.ndarray, names_csv: Path, content_csv: Path | None) -> list[dict]:
    """Display metadata for concept ids 1..C: English name, level, stage, difficulty, popularity."""
    ucids = concept_names[1:]
    names = pd.read_csv(names_csv).set_index("ucid").reindex(ucids)
    info = pd.read_csv(content_csv).set_index("ucid").reindex(ucids) if content_csv else None
    table = []
    for i, ucid in enumerate(ucids):
        row = names.iloc[i]
        entry = {
            "id": i + 1,
            "ucid": str(ucid),
            "name": row["name_en"] if isinstance(row["name_en"], str) else str(ucid),
            "level": row["level"] if isinstance(row["level"], str) else "",
            "popularity": int(popularity[i + 1]),
        }
        if info is not None:
            stage, difficulty = info.iloc[i]["learning_stage"], info.iloc[i]["difficulty"]
            entry["stage"] = stage if isinstance(stage, str) else ""
            entry["difficulty"] = difficulty if isinstance(difficulty, str) and difficulty != "unset" else ""
        table.append(entry)
    return table


def _summary(metrics_split: dict) -> dict:
    return {name: {k: round(v, 4) for k, v in m.items()} for name, m in metrics_split["metrics"].items()}


def export_bundle(
    data_dir: str | Path,
    runs_dir: str | Path,
    out_dir: str | Path,
    *,
    checkpoint: str = "pointer-seed0",
    names_csv: str | Path,
    content_csv: str | Path | None = None,
    split: str = "test",
    max_students: int = 300,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    data, runs, out = Path(data_dir), Path(runs_dir), Path(out_dir)
    proc = load_processed(data)
    with_examples = np.unique(proc.examples[split].student)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(with_examples, size=min(max_students, len(with_examples)), replace=False))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sub = subset_students(proc, split, chosen)
    save_processed(sub, out / "data")
    for src, dst in [
        (runs / checkpoint / "model.pt", "model.pt"),
        (runs / "dkt" / "dkt.pt", "dkt.pt"),
        (runs / "bkt" / "bkt_params.npz", "bkt_params.npz"),
        (runs / "prereq" / "prereq_graph.npz", "prereq_graph.npz"),
    ]:
        shutil.copyfile(src, out / dst)
    src, dst, counts = transition_counts(proc)
    np.savez_compressed(out / "markov.npz", src=src, dst=dst, counts=counts)
    concepts = concept_table(proc.inter.concept_names, proc.popularity, Path(names_csv), content_csv and Path(content_csv))
    save_json(concepts, out / "concepts.json")

    model_metrics = load_json(runs / checkpoint / "metrics.json")
    metrics = {"pointer": _summary(model_metrics["splits"][split])}
    baselines_file = runs / "baselines" / "baselines.json"
    if baselines_file.exists():
        for name, result in load_json(baselines_file)["baselines"].items():
            metrics[name] = _summary(result[split])
    dkt_summary = load_json(runs / "dkt" / "dkt_summary.json")
    date = dt.date.today().isoformat()
    meta = {
        "created": date,
        "source": {"data": data.name, "runs": runs.name, "checkpoint": checkpoint, "split": split},
        "dataset": DATASET,
        "num_concepts": proc.inter.num_concepts,
        "path_len": int(proc.examples[split].targets.shape[1]),
        "num_candidates": int(proc.examples[split].candidates.shape[1]),
        "students": int(len(chosen)),
        "examples": int(len(sub.examples[split])),
        "train_students": int(len(proc.splits["train"])),
        "metrics": metrics,
        "dkt_auc": dkt_summary.get("test", {}).get("auc"),
    }
    save_json(meta, out / "meta.json")
    (out / "README.md").write_text(
        README.format(data=data.name, runs=runs.name, date=date, students=len(chosen), **DATASET), encoding="utf-8"
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"exported {len(chosen)} students, {meta['examples']} examples to {out} ({size / 1e6:.1f} MB)")
    return meta
