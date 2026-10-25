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

from learnpath.baselines import transition_counts
from learnpath.preprocess import SPLITS, Examples, Interactions, Processed, load_processed, save_processed
from learnpath.utils import load_json, save_json

DATASET = {
    "name": "Junyi Academy Online Learning Activity Dataset",
    "url": "https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy",
    "license": "CC BY-NC-SA 4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}

README = """# Maths Learning Path Recommender app bundle

Exported by `learnpath export` from `{data}` and `{runs}` on {date}. See `meta.json` for
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
        student_uuid=np.array([f"Roll No. {i + 1:03d}" for i in range(len(students))]),
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


def topics_table(concept_names: np.ndarray, content_csv: Path, topics_csv: Path | None = None) -> dict:
    """Strands and units with English names, each unit with its concept ids (for the new-student questionnaire)."""
    from learnpath.onboarding import TOPICS_CSV, Topics

    topics = Topics.load(concept_names, content_csv, topics_csv or TOPICS_CSV)
    strand_index = {key: i for i, key in enumerate(topics.strands.id)}
    return {
        "strands": [
            {"id": i, "name": row.name_en, "stage": row.stage} for i, row in enumerate(topics.strands.itertuples())
        ],
        "units": [
            {"id": i, "strand": strand_index[row.parent], "name": row.name_en, "stage": row.stage,
             "concepts": row.concepts.tolist()}
            for i, row in enumerate(topics.units.itertuples())
        ],
    }


def _copy_results(runs: Path, results_dir: Path | None, out: Path) -> list[str]:
    """Copy the evaluation outputs that exist into ``out/results`` for the app's Results page."""
    sources = {"main.json": results_dir / "results.json" if results_dir else None}
    for f in sorted(runs.glob("catalogue_test_*.json")) + sorted(runs.glob("onboarding_*.json")):
        sources[f.name] = f
    for f in sorted((runs.parent / f"{runs.name}-wf").glob("report-*.json")):
        sources[f"walkforward_{f.stem[len('report-'):]}.json"] = f
    copied = []
    for name, src in sources.items():
        if src is not None and src.exists():
            (out / "results").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "results" / name)
            copied.append(name)
    return copied


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
    results_dir: str | Path | None = None,
    shortlist_model: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Write the app bundle (see the module docstring).

    Optional parts, each added when its input exists: the RL-fine-tuned
    policy for ``checkpoint`` (``rl.pt``), a model trained on Markov-shortlist
    distractors for the new-student page (``shortlist_model.pt``), the topic
    hierarchy (``topics.json``, needs ``content_csv``), and the evaluation
    outputs for the Results page (``results/``).
    """
    from learnpath.onboarding import curriculum_position

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
    position = curriculum_position(proc.inter, proc.splits["train"])
    for entry in concepts:
        entry["position"] = round(float(position[entry["id"]]), 4)
    if content_csv:
        topics = topics_table(proc.inter.concept_names, Path(content_csv))
        for unit in topics["units"]:
            for c in unit["concepts"]:
                concepts[c - 1]["unit"] = unit["id"]
        save_json(topics, out / "topics.json")
    save_json(concepts, out / "concepts.json")

    model_metrics = load_json(runs / checkpoint / "metrics.json")
    metrics = {"pointer": _summary(model_metrics["splits"][split])}
    rl_run = runs / checkpoint.replace("pointer", "rl", 1)
    rl = None
    if (rl_run / "model.pt").exists() and (rl_run / "metrics.json").exists():
        shutil.copyfile(rl_run / "model.pt", out / "rl.pt")
        rl_metrics = load_json(rl_run / "metrics.json")["splits"][split]
        metrics["rl"] = _summary(rl_metrics)
        rl = {"checkpoint": rl_run.name, "violations_per_path": rl_metrics["violations_per_path"]}
    if shortlist_model:
        shutil.copyfile(shortlist_model, out / "shortlist_model.pt")
    results = _copy_results(runs, Path(results_dir) if results_dir else None, out)
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
        "rl": rl,
        "shortlist_model": Path(shortlist_model).parent.name if shortlist_model else None,
        "results": results,
    }
    save_json(meta, out / "meta.json")
    (out / "README.md").write_text(
        README.format(data=data.name, runs=runs.name, date=date, students=len(chosen), **DATASET), encoding="utf-8"
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"exported {len(chosen)} students, {meta['examples']} examples to {out} ({size / 1e6:.1f} MB)")
    return meta
