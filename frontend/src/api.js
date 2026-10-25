// Same-origin by default: the API serves the built app, and Vite proxies /api in development.
const BASE = import.meta.env.VITE_API_URL ?? '';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // not JSON
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  info: () => request('/api/info'),
  students: () => request('/api/students'),
  student: (id) => request(`/api/students/${id}`),
  example: (id) => request(`/api/examples/${id}`),
  simulate: (id) => request(`/api/examples/${id}/simulate`),
  graph: (id) => request(`/api/examples/${id}/graph`),
  tutor: (id, body) => request(`/api/examples/${id}/tutor`, { method: 'POST', body: JSON.stringify(body) }),
  questionnaire: () => request('/api/questionnaire'),
  plan: (body) => request('/api/plan', { method: 'POST', body: JSON.stringify(body) }),
  explain: (body) => request('/api/plan/explain', { method: 'POST', body: JSON.stringify(body) }),
  results: () => request('/api/results'),
};

// Series colours for the plan and comparison charts (the dataviz skill's first three validated slots).
export const PLAN_COLOR = { pointer: 'var(--series-1)', rl: 'var(--series-3)', markov: 'var(--series-2)' };

// Categorical slots in fixed order; "actual" is a reference line, not a series colour.
export const METHOD_COLOR = {
  pointer: 'var(--series-1)',
  markov: 'var(--series-2)',
  popularity: 'var(--series-3)',
  bkt_gain: 'var(--series-4)',
  actual: 'var(--reference)',
};

export const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;

// Indian conventions for everything the app displays: lakh and crore digit grouping (1,62,17,311),
// day-month-year dates, and Junyi's school stages (Taiwan's grades 1-6, 7-9 and 10-12) named by the
// Indian classes they roughly match.
export const fmtNum = (v) => Number(v).toLocaleString('en-IN');
export const fmtDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
export const STAGE_LABEL = {
  elementary: 'Primary (Class 1–6)',
  junior: 'Middle & Secondary (Class 7–9)',
  senior: 'Senior Secondary (Class 10–12)',
};
export const stageLabel = (stage) => STAGE_LABEL[stage] ?? stage;
