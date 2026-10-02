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
.venv/bin/python -m pytest          # 55 tests, about 45 s
```

Training on the full dataset needs a GPU. On a 4-core CPU the pointer network takes about
4 ms per training example per epoch, so the full grid runs on Colab: open
[`notebooks/pathtrace_colab.ipynb`](notebooks/pathtrace_colab.ipynb) in Colab with a GPU
runtime and run all cells. It keeps data and runs on Google Drive and resumes after a
disconnect.

## Data

The [Junyi Academy Online Learning Activity dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
is a public download (1.6 GB zip, no Kaggle account needed):

```bash
mkdir -p data/raw/junyi && cd data/raw/junyi
curl -L -o junyi.zip https://www.kaggle.com/api/v1/datasets/download/junyiacademy/learning-activity-public-dataset-by-junyi-academy
unzip junyi.zip && rm junyi.zip    # Log_Problem.csv (3.0 GB), Info_Content.csv, Info_UserData.csv
```

`ml/data/` and `ml/runs/` are git-ignored, so raw data and run outputs never get committed.

## The whole grid in two commands

```bash
python -m pathtrace grid --data data/processed/junyi --runs runs/junyi --content data/raw/junyi/Info_Content.csv
python -m pathtrace report --data data/processed/junyi --runs runs/junyi --out results/junyi
```

`grid` runs every step below for one dataset (simulators, 4 baselines, 3 model variants x 5
seeds, RL x 5 seeds, 3 RL sensitivity runs, DKT learning gain) and skips any step whose
output already exists. `report` turns the run folders into the paper's main table
(`main_table.md`, `.csv`, `.tex`: mean ± std over seeds, BKT reward as a share of the oracle,
prerequisite violations, DKT gain, and the NDCG@3 difference to the strongest baseline with a
paired bootstrap CI and p-value) and figures (`fig_training.png`, `fig_rl.png`,
`fig_methods.png`).

## Individual commands

```bash
# 1. Raw log -> student-level 70/15/15 split -> sliding-window examples
python -m pathtrace prepare --log data/raw/junyi/Log_Problem.csv --out data/processed/junyi

# 2. Baselines: random, popularity, markov, gru_next_item (val + test, bootstrap CIs)
python -m pathtrace baselines --data data/processed/junyi --out runs/junyi/baselines

# 3. Pointer network: supervised training with early stopping, then val + test
python -m pathtrace train --data data/processed/junyi --out runs/junyi/pointer-seed0 --seed 0

# 4. Student simulators and structure, all fitted on the train split
python -m pathtrace bkt --data data/processed/junyi --out runs/junyi/bkt          # RL reward (per-concept BKT)
python -m pathtrace prereq --data data/processed/junyi --out runs/junyi/prereq \
    --content data/raw/junyi/Info_Content.csv                                      # prerequisite graph
python -m pathtrace dkt --data data/processed/junyi --out runs/junyi/dkt          # held-out evaluator

# 5. RL fine-tuning of a supervised checkpoint against the BKT reward
python -m pathtrace finetune --data data/processed/junyi --checkpoint runs/junyi/pointer-seed0/model.pt \
    --bkt runs/junyi/bkt/bkt_params.npz --prereq runs/junyi/prereq/prereq_graph.npz --out runs/junyi/rl-seed0

# 6. DKT learning gain of every method's paths (plus random, the students' own, and the BKT oracle)
python -m pathtrace gain --data data/processed/junyi --dkt runs/junyi/dkt/dkt.pt \
    --runs runs/junyi/baselines runs/junyi/pointer-seed0 runs/junyi/rl-seed0 --bkt runs/junyi/bkt/bkt_params.npz

# 7. Re-evaluate a checkpoint
python -m pathtrace evaluate --data data/processed/junyi --checkpoint runs/junyi/pointer-seed0/model.pt --split test
```

`python -m pathtrace <command> --help` lists every option with its default. When running
several jobs on one CPU, give each `--threads 1` (before the command name): oversubscribed
PyTorch threads made two jobs on 4 cores run about 30x slower.

| Experiment | Flags |
| --- | --- |
| Paper's original setup (one last-K example per student) | `prepare --windows last` |
| Harder candidate sets | `prepare --negatives popularity` |
| Quick run on a subsample | `prepare --max-students 5000` |
| Without the KT head | `train --kt-weight 0` |
| Without the candidate Transformer | `train --attn-layers 0` |
| Another seed | `train --seed 1` |
| RL without the supervised anchor | `finetune --sup-weight 0` |
| RL with the moving-average baseline | `finetune --baseline ema` |
| No prerequisite penalty | `finetune --violation-weight 0` |

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

- **Section 4.6 and 8(3), prerequisites.** The prerequisite map is no longer empty: it is
  inferred from training students' first-attempt order (see `pathtrace/prereq.py`), and it
  agrees with Junyi's own topic hierarchy and learning stages far above chance.
- **Section 5.2 and Algorithm 1, RL.** The baseline is the greedy path's reward
  (self-critical), the entropy bonus is applied, a supervised term (0.5) anchors the policy,
  and training runs up to 20 epochs with early stopping on validation reward. The
  moving-average baseline is kept as an ablation, now applied before its update.
- **New evaluation.** DKT learning gain E_p (`pathtrace/dkt.py`) and the share of the BKT
  oracle's reward, alongside the ranking metrics.

- **Section 5.1, batch size.** The Colab grid uses batch size 256 to keep the GPU busy; report
  that value for every number that comes from it.
