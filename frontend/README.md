# PathTrace frontend

React + Vite + Tailwind app for the PathTrace demo. See the [root README](../README.md) to
run it with the API.

- `npm run build`: production build into `dist/`, which `backend/server.py` serves.
- `npm run dev`: dev server on port 5173; `/api` is proxied to `http://127.0.0.1:8000`.
- `npm run lint`: oxlint.

Set `VITE_API_URL` at build time to call an API on another origin.
