# Running the experiments

How to reproduce every result in `ml/results/` and the app bundle. To install and run the app,
see the [README](README.md); the steps below use the same virtual environment.

| What | Hardware | Time |
| --- | --- | --- |
| [1. Full experiments on a local GPU](#1-full-experiments-on-a-local-gpu) | NVIDIA GPU, 16 GB RAM, about 10 GB free disk | Several hours; resumable |
| [2. The ML code on a synthetic log](#2-the-ml-code-on-a-synthetic-log) | Any CPU | About a minute |

## 1. Full experiments on a local GPU

One command runs everything: it downloads Junyi (1.6 GB, no Kaggle account), prepares the
datasets, runs the experiment grids (supervised training, then the RL stage, which always
runs), writes the tables and figures, and exports the app bundle.

1. With the virtual environment active, replace the CPU build of PyTorch with the CUDA one.
   RTX 50-series cards need CUDA 12.8 or newer; `cu130` works on any recent NVIDIA driver:

   ```bash
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   The last line should print `True` and the GPU's name.
2. From the `ml/` folder, start the run and send its output to a log file:

   ```bash
   cd ml
   python -m learnpath all --seeds 0 1 2 > all.log 2>&1
   ```

   Watch progress with `Get-Content all.log -Wait` (PowerShell) or `tail -f all.log`.

- **Stopping and resuming.** Stop it with Ctrl+C, or restart the machine; run the same command
  again and it continues. Finished steps are skipped, and a training run that was cut off
  continues from its last finished epoch with the same result it would have had.
- **Sleep.** The run takes hours, so keep the machine from sleeping until it finishes (on
  Windows: Settings → System → Power).
- **Order.** The main dataset (`junyi`) finishes first: its grid, then `results/junyi/` and
  the app bundle in `ml/bundle/`. The two optional grids run after it. To skip them, add
  `--datasets junyi`. For 5 seeds, run again later with `--seeds 0 1 2 3 4`: the first three
  are already done.
- **Results** are in `ml/results/<dataset>/`:
  - `main_table.md`, `main_table.csv` and `main_table.tex`: mean ± std over seeds, with the
    paired bootstrap comparison against the strongest baseline;
  - `fig_methods.png`: every method on NDCG@3, with confidence intervals;
  - `fig_training.png`: training curves;
  - `fig_rl.png`: the RL reward and validation curves.

  The same table and figures for an earlier 10,000-student subsample (one seed, trained on a
  CPU) are in [`ml/results/junyi-10k`](ml/results/junyi-10k), for comparison.
- **Use the new models in the app:** replace `backend/bundle` with `ml/bundle` and restart
  the server. That bundle also uses the main model on the *Plan my next topics* page. The
  committed bundle's Plan-page model (trained on realistic distractors, then RL-tuned) comes
  from the commands under *Testing on genuinely unseen data* in [`ml/README.md`](ml/README.md).

## 2. The ML code on a synthetic log

With the virtual environment active, from the `ml/` folder. This uses a synthetic
Junyi-shaped log, so it runs in about a minute on a CPU; the numbers are not real results.

```bash
cd ml
python -m learnpath synth --out demo/Log_Problem.csv --students 400 --concepts 80 --mean-length 80
python -m learnpath prepare --log demo/Log_Problem.csv --out demo/processed
python -m learnpath baselines --data demo/processed --out demo/runs/baselines --only random,popularity,markov
python -m learnpath train --data demo/processed --out demo/runs/pointer-seed0 --max-epochs 5
```

`train` prints the loss and validation NDCG@3 every epoch, then val and test tables. Delete the
`demo/` folder afterwards.

To run on the real data instead, follow [`ml/README.md`](ml/README.md): it downloads the
dataset and lists every command. Training on a CPU is slow (about 3 minutes per epoch on a
10,000-student subsample with 4 cores), which is why the full runs need a GPU.
