# PathTrace

Learning-path recommendation on real Junyi Academy students. A pointer network reads a
student's attempt history and picks the next 3 concepts, in order, from a candidate set. It
is compared against Popularity, Markov and GRU next-item baselines, and scored on a held-out
DKT student simulator.

| Folder | What it is |
| --- | --- |
| [`ml/`](ml/README.md) | The `pathtrace` package: data pipeline, model, baselines, simulators, RL fine-tuning, experiment grid and paper tables |
| [`backend/`](backend/server.py) | FastAPI service that serves the trained model on held-out students |
| [`frontend/`](frontend/) | React app: recommended path, attention, baselines, simulation, prerequisite map and tutor |

## Run the demo app

Needs Python 3.10+ and Node 20+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ml -r backend/requirements.txt
(cd frontend && npm ci && npm run build)

python backend/server.py               # open http://localhost:8000
```

The server loads `backend/bundle` (a trained checkpoint, its simulators and 300 held-out test
students, 3.2 MB) and serves the built frontend. Nothing else needs downloading.

- **Tutor.** It works without a key, answering from a template. For Gemini answers, copy
  `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY`, or paste a key in the app.
- **Frontend development.** Run `npm run dev` in `frontend/` alongside the server; Vite
  proxies `/api` to port 8000.
- **Tests.** `pytest backend/tests` and `pytest ml`.

To serve a different model, export a new bundle with `python -m pathtrace export` (see
[`ml/README.md`](ml/README.md)) and point `PATHTRACE_BUNDLE` at it.

## Data

[Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy),
licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The bundled
student histories and the English concept names in `ml/resources/` are derived from it and
shared under the same license. The concept names are machine-translated.
