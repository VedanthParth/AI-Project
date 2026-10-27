# Maths Learning Path Recommender

Learning-path recommendation on real Junyi Academy students. A pointer network reads a
student's attempt history and picks the next 3 concepts, in order, from a candidate set. It is
trained on what students really did next, then fine-tuned with reinforcement learning for expected
learning and for not skipping prerequisites; the app serves only the fine-tuned model. It is
compared against Popularity, Markov and GRU next-item baselines, and scored on a held-out
DKT student simulator. Beyond the main held-out split it is tested on the whole catalogue (no
candidate list that holds the answer), walk-forward in time, and on new students described
only by a short questionnaire.

| Folder | What it is |
| --- | --- |
| [`ml/`](ml/README.md) | The `learnpath` package: data pipeline, model, baselines, simulators, RL fine-tuning, experiment grid, evaluations and paper tables |
| [`backend/`](backend/server.py) | FastAPI service that serves the trained models |
| [`frontend/`](frontend/) | React app for students and parents: plan your next topics from three questions, see it on real held-out students, and how it works, with spider and line charts explaining every suggestion |

## Run the app

The app serves the trained models on 300 held-out Junyi students. Everything it needs is in
`backend/bundle` (5 MB), so nothing else needs downloading.

You need [Python 3.10 or newer](https://www.python.org/downloads/) (3.12 recommended) and
[Node.js 20.19 or newer](https://nodejs.org/en/download).

### Get the code

```bash
git clone --branch pathtrace-rebuild https://github.com/VedanthParth/AI-Project.git
cd AI-Project
```

No git? Download the [branch as a zip](https://github.com/VedanthParth/AI-Project/archive/refs/heads/pathtrace-rebuild.zip)
and unzip it.

### Install and start

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
  `backend/.env`, set `GEMINI_API_KEY` and restart the server. The tutor shows *Connected to Gemini*
  when the key is loaded.
- **Links to a view.** The address bar keeps the page, student and moment, so a view can be
  reopened or shared later.
- **Frontend development.** Run `npm run dev` in `frontend/` alongside the server; Vite
  proxies `/api` to port 8000.
- **Tests.** `python -m pip install pytest` once, then `python -m pytest` in `ml/` and
  `python -m pytest backend/tests` from the repository root.
- **Another model.** Export a bundle with `python -m learnpath export` (see
  [`ml/README.md`](ml/README.md)) and point `LEARNPATH_BUNDLE` at it. Export accepts only
  RL-fine-tuned checkpoints.

## Reproduce the results

[RUNNING.md](RUNNING.md) runs the full experiments on a local NVIDIA GPU, and the ML code on a
small synthetic log. [`ml/README.md`](ml/README.md) lists every command and option.

## Data

[Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy),
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The bundled
student histories and the English concept names in `ml/resources/` are derived from it and
shared under the same license. The concept names are machine-translated.
