# How to run the Maths Learning Path Recommender

Everything lives on the `pathtrace-rebuild` branch:
https://github.com/VedanthParth/AI-Project/tree/pathtrace-rebuild

| What | Where it runs | Time |
| --- | --- | --- |
| [1. Demo app](#1-run-the-demo-app) | Your laptop | 10 min to install, then instant |
| [2. Full experiments](#2-run-the-full-experiments-on-your-own-gpu) | Your laptop's GPU, or [Colab](#2b-or-run-them-on-colab) | Several hours; resumable |
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
  http://localhost:8000/?tab=students&student=43&moment=472.

## 2. Run the full experiments on your own GPU

One command does everything the Colab notebook does: download Junyi (1.6 GB, no Kaggle
account), prepare the datasets, run the experiment grids, write the tables and figures, and
export the app bundle. It needs an NVIDIA GPU, about 10 GB of free disk and 16 GB of RAM.

1. With the virtual environment from step 1 active, replace the CPU build of PyTorch with
   the CUDA one. RTX 50-series cards need CUDA 12.8 or newer; `cu130` works on any recent
   NVIDIA driver:

   ```bash
   pip uninstall -y torch
   pip install torch --index-url https://download.pytorch.org/whl/cu130
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   The last line should print `True` and your GPU's name.
2. From the `ml/` folder, start the run and send its output to a log file:

   ```bash
   cd ml
   python -m learnpath all --seeds 0 1 2 > all.log 2>&1
   ```

   Watch progress with `Get-Content all.log -Wait` (PowerShell) or `tail -f all.log`.

- **Stopping and resuming.** Stop it with Ctrl+C, or let the laptop restart; run the same
  command again and it continues. Finished steps are skipped, and a training run that was
  cut off continues from its last finished epoch with the same result it would have had.
- **Keep the laptop plugged in and awake.** Windows: Settings → System → Power → set *When
  plugged in, put my device to sleep after* to *Never* for the run.
- **Order.** The main dataset (`junyi`) finishes first: its grid, then `results/junyi/` and
  the app bundle in `ml/bundle/`. The two optional grids run after it. To skip them, add
  `--datasets junyi`. For the paper's 5 seeds, run again later with `--seeds 0 1 2 3 4`:
  the first three are already done.
- **Results** are in `ml/results/<dataset>/` (see the list under Colab below).
- **Put the full-data model in the app:** delete `backend/bundle` and copy `ml/bundle` in its
  place, then restart `python backend/server.py`.

## 2b. Or run them on Colab

The same grid, on a Colab GPU. Free sessions disconnect, so this is the fallback.

1. Open the notebook in Colab:
   **[colab_experiments.ipynb](https://colab.research.google.com/github/VedanthParth/AI-Project/blob/pathtrace-rebuild/ml/notebooks/colab_experiments.ipynb)**
   - The notebook is only on the `pathtrace-rebuild` branch, at `ml/notebooks/colab_experiments.ipynb`,
     so it doesn't show up on GitHub's default `master` view.
   - If the link doesn't open: in Colab, use **File → Open notebook → GitHub**, enter
     `VedanthParth/AI-Project`, choose the `pathtrace-rebuild` branch, and pick
     `ml/notebooks/colab_experiments.ipynb`. Or use **File → Upload notebook** with
     `ml\notebooks\colab_experiments.ipynb` from your clone.
2. **Runtime → Change runtime type → T4 GPU** (or any GPU), then **Save**.
3. **Runtime → Run all.** Allow access to Google Drive when asked.

What each section does:

| Section | What it does | Notes |
| --- | --- | --- |
| 1. Drive, code and dependencies | Mounts Drive, clones this branch, installs `ml/` | Uses `MyDrive/learnpath` for everything |
| 2. Data | Downloads Junyi (1.6 GB, no Kaggle account) and prepares 3 datasets | Download happens once |
| 3. Experiment grids | Simulators, baselines, pointer network ×3 variants × seeds, RL, DKT gain | The long step. The last two grids are optional |
| 4. Tables and figures | Writes the results and shows them in the notebook | Screenshot the figures here |
| 5. App bundle | Exports the full-data model for the app as `bundle.zip` | See below |

- **If Colab disconnects** (free sessions time out and stop when idle), reconnect and
  **Run all** again. Every finished step is skipped, and an interrupted training run continues
  from its last finished epoch.
- **Short on GPU time?** In the cell that sets `SEEDS`, use `SEEDS = '0 1 2'`, and skip the two
  cells marked *Optional*. Each training run prints its seconds per epoch, so the first run
  tells you how long the rest will take.
- **Results** are in Drive under `MyDrive/learnpath/results/junyi/`:
  - `main_table.md`, `main_table.csv` and `main_table.tex`: mean ± std over seeds, with the
    paired bootstrap comparison against the strongest baseline;
  - `fig_methods.png`: every method on NDCG@3, with confidence intervals;
  - `fig_training.png`: training curves;
  - `fig_rl.png`: the RL reward and validation curves.
- **Put the full-data model in the app:** download `MyDrive/learnpath/bundle.zip`, unzip it,
  delete the repo's `backend/bundle` folder and put the unzipped `bundle` folder in its place.
  Restart `python backend/server.py`.

## 3. Run the ML code locally

With the virtual environment from step 1 active, from the `ml/` folder. This uses a
synthetic Junyi-shaped log, so it runs in about a minute on a laptop; the numbers are not
real results.

```bash
cd ml
python -m learnpath synth --out demo/Log_Problem.csv --students 400 --concepts 80 --mean-length 80
python -m learnpath prepare --log demo/Log_Problem.csv --out demo/processed
python -m learnpath baselines --data demo/processed --out demo/runs/baselines --only random,popularity,markov
python -m learnpath train --data demo/processed --out demo/runs/pointer-seed0 --max-epochs 5
```

`train` prints the loss and validation NDCG@3 every epoch, then val and test tables. Delete the
`demo/` folder afterwards.

Tests, with the virtual environment active: `python -m pip install pytest` once, then
`python -m pytest` in `ml/` (93 tests, about 1 minute) and `python -m pytest backend/tests` from the
repository root (10 tests).

To run on the real data instead, follow [`ml/README.md`](ml/README.md): it downloads the
dataset and lists every command. Training on a CPU is slow (about 3 minutes per epoch on a
10,000-student subsample with 4 cores), which is why the full runs need a GPU (section 2).

## 4. Live demo and screenshot checklist

**Live demo, about 3 minutes** (http://localhost:8000):

1. **New student** tab, click **Example**: a Class 7–9 student who is average at Algebra,
   confident in Linear Equations in One Variable, needs help with Quadratic Equations, and last
   finished an early Quadratic Equations exercise.
2. **How the path is chosen**: from all 1,326 exercises to the class group, to what is not done
   yet, to the Markov shortlist, to the 3 the pointer network orders.
3. **Recommended path**: each step's attention over the shortlist, and whether its inferred
   prerequisites are covered. **Three ways to choose** puts it next to Markov and the RL-tuned
   policy on the same shortlist.
4. Click **Struggled** on step 1: the banner says how the path changed, and **Review first**
   lists the exercise (a mastery rule, separate from the models).
5. **Real students** tab: a held-out student, where what they really did next is known.
6. **Results** tab: how well it works on unseen students, including the whole-catalogue,
   questionnaire and walk-forward results, and where it does not beat Markov.

**Screenshots** (Real students tab unless noted):

1. The whole page on one student (browser zoomed out to about 67%).
2. **Recommended path**: the three steps with attention, mastery and outcome.
3. **All 20 candidates and the model's attention**: click *Show all 20*.
4. **Compare with the baselines** table.
5. **Simulate following each path** chart, with the mouse over a point to show the tooltip.
6. **Prerequisite map**, both *Around the path* and *All linked candidates*
   (`?tab=students&student=43&moment=472` with the committed bundle).
7. **Ask the tutor** after clicking *Why this path?*.
8. **About this model** table at the bottom.
9. **New student** tab after *Example*, then again after *Struggled* on step 1.
10. **Results** tab.

**Results** (`ml/results/junyi/` after section 2, or `MyDrive/learnpath/results/junyi/` on Colab):

11. The main results table.
12. `fig_methods.png`, `fig_training.png` and `fig_rl.png`.

**Terminal** (section 3 above):

13. The `train` output, showing epochs and the val/test tables.
14. `python -m pytest` passing.
