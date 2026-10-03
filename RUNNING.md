# How to run PathTrace

Everything lives on the `pathtrace-rebuild` branch:
https://github.com/VedanthParth/AI-Project/tree/pathtrace-rebuild

| What | Where it runs | Time |
| --- | --- | --- |
| [1. Demo app](#1-run-the-demo-app) | Your laptop | 10 min to install, then instant |
| [2. Full experiments](#2-run-the-full-experiments-on-colab) | Google Colab (GPU) | Several hours; resumable |
| [3. ML code in a terminal](#3-run-the-ml-code-locally) | Your laptop | About 1 minute on synthetic data |
| [4. Screenshot checklist](#4-screenshot-checklist-for-the-presentation) | | |

## 0. Get the code

```bash
git clone --branch pathtrace-rebuild https://github.com/VedanthParth/AI-Project.git
cd AI-Project
```

No git? Download the [branch as a zip](https://github.com/VedanthParth/AI-Project/archive/refs/heads/pathtrace-rebuild.zip)
and unzip it.

You need [Python 3.10 or newer](https://www.python.org/downloads/) (3.12 recommended) and
[Node.js 20.19 or newer](https://nodejs.org/en/download).

## 1. Run the demo app

The app serves the trained model on 300 held-out Junyi students. Everything it needs is in
`backend/bundle`, so there is nothing to download.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ml -r backend/requirements.txt
cd frontend && npm ci && npm run build && cd ..
python backend/server.py
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ml -r backend/requirements.txt
cd frontend; npm ci; npm run build; cd ..
python backend/server.py
```

Then open **http://localhost:8000**. Stop the server with Ctrl+C. Next time, only the
`activate` line and `python backend/server.py` are needed.

- **Linux download size.** `pip` fetches the CUDA build of PyTorch (about 2.5 GB) by default.
  The app only needs the CPU build: run
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the other `pip` line.
- **PowerShell refuses to activate** the virtual environment: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
- **Tutor.** It answers from a template without a key. For Gemini answers, get a key from
  [Google AI Studio](https://aistudio.google.com/apikey), copy `backend/.env.example` to
  `backend/.env` and set `GEMINI_API_KEY`, or paste the key under "Gemini API key" in the app.
- **Links to a specific moment.** The address bar keeps the student and moment, so a view can
  be reopened later. With the committed bundle, a good one for the prerequisite map is
  http://localhost:8000/?student=112&moment=1200.

## 2. Run the full experiments on Colab

1. Open the notebook in Colab:
   **[pathtrace_colab.ipynb](https://colab.research.google.com/github/VedanthParth/AI-Project/blob/pathtrace-rebuild/ml/notebooks/pathtrace_colab.ipynb)**
2. **Runtime → Change runtime type → T4 GPU** (or any GPU), then **Save**.
3. **Runtime → Run all.** Allow access to Google Drive when asked.

What each section does:

| Section | What it does | Notes |
| --- | --- | --- |
| 1. Drive, code and dependencies | Mounts Drive, clones this branch, installs `ml/` | Uses `MyDrive/pathtrace` for everything |
| 2. Data | Downloads Junyi (1.6 GB, no Kaggle account) and prepares 3 datasets | Download happens once |
| 3. Experiment grids | Simulators, baselines, pointer network ×3 variants × seeds, RL, DKT gain | The long step. The last two grids are optional |
| 4. Tables and figures | Writes the results and shows them in the notebook | Screenshot the figures here |
| 5. App bundle | Exports the full-data model for the app as `bundle.zip` | See below |

- **If Colab disconnects** (free sessions time out and stop when idle), reconnect and
  **Run all** again. Every finished step is skipped, so it continues where it stopped.
- **Short on GPU time?** In the cell that sets `SEEDS`, use `SEEDS = '0 1 2'`, and skip the two
  cells marked *Optional*. Each training run prints its seconds per epoch, so the first run
  tells you how long the rest will take.
- **Results** are in Drive under `MyDrive/pathtrace/results/junyi/`:
  - `main_table.md`, `main_table.csv` and `main_table.tex`: mean ± std over seeds, with the
    paired bootstrap comparison against the strongest baseline;
  - `fig_methods.png`: every method on NDCG@3, with confidence intervals;
  - `fig_training.png`: training curves;
  - `fig_rl.png`: the RL reward and validation curves.
- **Put the full-data model in the app:** download `MyDrive/pathtrace/bundle.zip`, unzip it,
  delete the repo's `backend/bundle` folder and put the unzipped `bundle` folder in its place.
  Restart `python backend/server.py`.

## 3. Run the ML code locally

With the virtual environment from step 1 active, from the `ml/` folder. This uses a
synthetic Junyi-shaped log, so it runs in about a minute on a laptop; the numbers are not
real results.

```bash
cd ml
python -m pathtrace synth --out demo/Log_Problem.csv --students 400 --concepts 80 --mean-length 80
python -m pathtrace prepare --log demo/Log_Problem.csv --out demo/processed
python -m pathtrace baselines --data demo/processed --out demo/runs/baselines --only random,popularity,markov
python -m pathtrace train --data demo/processed --out demo/runs/pointer-seed0 --max-epochs 5
```

`train` prints the loss and validation NDCG@3 every epoch, then val and test tables. Delete the
`demo/` folder afterwards.

Tests: `pytest` in `ml/` (68 tests, about 30 s) and `pytest backend/tests` from the repository
root (6 tests).

To run on the real data instead, follow [`ml/README.md`](ml/README.md): it downloads the
dataset and lists every command. Training on a CPU is slow (about 3 minutes per epoch on a
10,000-student subsample with 4 cores), which is why the full runs use Colab.

## 4. Screenshot checklist for the presentation

**Demo app** (http://localhost:8000):

1. The whole page on one student (browser zoomed out to about 67%).
2. **Recommended path**: the three steps with attention, mastery and outcome.
3. **All 20 candidates and the model's attention**: click *Show all 20*.
4. **Compare with the baselines** table.
5. **Simulate following each path** chart, with the mouse over a point to show the tooltip.
6. **Prerequisite map**, both *Around the path* and *All linked candidates*
   (`?student=112&moment=1200` with the committed bundle).
7. **Ask the tutor** after clicking *Why this path?*.
8. **About this model** table at the bottom.

**Colab** (section 4 of the notebook):

9. The main results table.
10. `fig_methods.png`, `fig_training.png` and `fig_rl.png`.

**Terminal** (section 3 above):

11. The `train` output, showing epochs and the val/test tables.
12. `pytest` passing.
