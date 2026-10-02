# PathTrace ML pipeline

Learning path recommendation on the Junyi Academy dataset: a GRU student encoder,
a Transformer encoder over the candidate set, an LSTM pointer decoder and an
auxiliary knowledge-tracing head (paper Section 4), plus the baselines it must beat.

Every number reported in the paper should come from these commands.

## Setup

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest          # 38 tests, about 15 s
```

Training on the full dataset needs a GPU (Colab is enough). On a 4-core CPU the
pointer network takes about 4 ms per training example per epoch.

## Data

The [Junyi Academy Online Learning Activity dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
is a public download (1.6 GB zip, no Kaggle account needed):

```bash
mkdir -p data/raw/junyi && cd data/raw/junyi
curl -L -o junyi.zip https://www.kaggle.com/api/v1/datasets/download/junyiacademy/learning-activity-public-dataset-by-junyi-academy
unzip junyi.zip && rm junyi.zip    # Log_Problem.csv (3.0 GB), Info_Content.csv, Info_UserData.csv
```

`ml/data/` and `ml/runs/` are git-ignored, so raw data and run outputs never get committed.

## Commands

```bash
# 1. Raw log -> student-level 70/15/15 split -> sliding-window examples
python -m pathtrace prepare --log data/raw/junyi/Log_Problem.csv --out data/processed/junyi

# 2. Baselines: random, popularity, markov, gru_next_item (val + test, bootstrap CIs)
python -m pathtrace baselines --data data/processed/junyi --out runs/junyi/baselines

# 3. Pointer network: supervised training with early stopping, then val + test
python -m pathtrace train --data data/processed/junyi --out runs/junyi/pointer-seed0 --seed 0

# 4. Per-concept BKT (the RL reward's student simulator), fitted on the train split
python -m pathtrace bkt --data data/processed/junyi --out runs/junyi/bkt

# 5. Re-evaluate a checkpoint
python -m pathtrace evaluate --data data/processed/junyi --checkpoint runs/junyi/pointer-seed0/model.pt --split test
```

`python -m pathtrace <command> --help` lists every option with its default.

| Experiment | Flags |
| --- | --- |
| Paper's original setup (one last-K example per student) | `prepare --windows last` |
| Harder candidate sets | `prepare --negatives popularity` |
| Quick run on a subsample | `prepare --max-students 5000` |
| Without the KT head | `train --kt-weight 0` |
| Without the candidate Transformer | `train --attn-layers 0` |
| Another seed | `train --seed 1` |

Each run directory holds `model.pt` (checkpoint), `history.json` (per-epoch train and
validation losses and metrics, for the training-curve figure), `metrics.json`
(val/test metrics with 95% bootstrap CIs, KT-head AUC, and the analytic chance level)
and `preds_<split>.npy` (predicted paths, for significance tests).

## Development without the real data

```bash
python -m pathtrace synth --out data/raw/synthetic/Log_Problem.csv
python -m pathtrace prepare --log data/raw/synthetic/Log_Problem.csv --out data/processed/synthetic
```

The synthetic log has the same columns as `Log_Problem.csv`. **Its numbers mean
nothing and must never appear in the paper.**

## Where the method differs from the current paper text

The Paper lane should update these sections to match the code:

- **Section 6.1, ordering.** Junyi rounds `timestamp_TW` to 15-minute windows, and
  88.6% of a student's consecutive attempts share a timestamp. The order of attempts
  at different concepts inside one window is therefore unknown. Attempts are sorted by
  window, then concept, then `exercise_problem_repeat_session` and `problem_number`
  (which order a concept's own attempts exactly). Cuts fall only on window boundaries,
  concepts started in the same window as the last target are never negatives, and
  `first_step` accepts any target from the first window.

- **Section 6.2, examples.** One example per cut point in each student's sequence of
  new concepts (capped at 20 per student), not one per student. The history is every
  attempt before the cut; the target is the next 3 new concepts. `--windows last`
  reproduces the original setup.
- **Section 6.2, negatives.** Negatives never include concepts already in the history
  (`--exclude-seen`). Otherwise "never pick a concept from the history" would be a free
  shortcut, because targets are always new concepts.
- **Section 4.4, decoder order.** The LSTM step comes first, then attention:
  `h_t = LSTM([e_prev; context_{t-1}], h_{t-1})`, then the scores over eligible
  candidates, then `context_t`. `h_0 = s`, and `e_prev` starts as a learned vector.
  See the docstring in `pathtrace/model.py`.
- **Section 5.1, optimisation.** Batch size 64, weight decay 1e-5, dropout 0.1, and
  early stopping on validation NDCG@3 (patience 5, at most 50 epochs). Histories are
  truncated to their last 100 attempts.
- **Section 6.4, metrics.** Adds `first_step` (is the first recommended concept the
  one the student attempted next?), the only order-sensitive metric. Precision@3
  equals Recall@3 by construction, so report one of them.

- **Section 4.6, BKT.** Parameters are fitted per concept by EM on the training split
  (guess and slip capped at 0.3; concepts with fewer than 50 attempts use a global fit),
  not fixed at 0.30/0.15/0.20/0.10. The reward is the exact expected gain
  (1 - m) * p_learn rather than a sampled outcome. With the paper's single parameter
  set, every unseen candidate has the same expected gain, 0.105.

Not built yet: the DKT evaluator, the inferred prerequisite graph, the corrected REINFORCE
fine-tuning, the `report` command and the Colab notebook.
