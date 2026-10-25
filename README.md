# Maths Learning Path Recommender

Learning-path recommendation on real Junyi Academy students. A pointer network reads a
student's attempt history and picks the next 3 concepts, in order, from a candidate set. It
is compared against Popularity, Markov and GRU next-item baselines, and scored on a held-out
DKT student simulator. Beyond the main held-out split it is tested on the whole catalogue (no
candidate list that holds the answer), walk-forward in time, and on new students described
only by a short questionnaire.

| Folder | What it is |
| --- | --- |
| [`ml/`](ml/README.md) | The `learnpath` package: data pipeline, model, baselines, simulators, RL fine-tuning, experiment grid, evaluations and paper tables |
| [`backend/`](backend/server.py) | FastAPI service that serves the trained models |
| [`frontend/`](frontend/) | React app: a new student's path built step by step from a questionnaire, held-out students with what they really did, and the results |

**[RUNNING.md](RUNNING.md)** has step-by-step instructions for everything: the app on Windows,
macOS or Linux, the Colab experiments, and the ML code in a terminal.

## Run the demo app

Needs Python 3.10+ and Node 20.19+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ml -r backend/requirements.txt
(cd frontend && npm ci && npm run build)

python backend/server.py               # open http://localhost:8000
```

The server loads `backend/bundle` (the full-data checkpoints, their simulators, 300 held-out
test students and the evaluation results, 5 MB) and serves the built frontend. Nothing else
needs downloading.

- **Tutor.** It works without a key, answering from a template. For Gemini answers, copy
  `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY`, or paste a key in the app.
- **Frontend development.** Run `npm run dev` in `frontend/` alongside the server; Vite
  proxies `/api` to port 8000.
- **Tests.** `python -m pip install pytest`, then `python -m pytest backend/tests` and `python -m pytest ml`.

To serve a different model, export a new bundle with `python -m learnpath export` (see
[`ml/README.md`](ml/README.md)) and point `LEARNPATH_BUNDLE` at it.

## Data

[Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy),
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The bundled
student histories and the English concept names in `ml/resources/` are derived from it and
shared under the same license. The concept names are machine-translated.
