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
};

export const METHOD_ORDER = ['pointer', 'markov', 'popularity', 'bkt_gain', 'actual'];

// Categorical slots in fixed order; "actual" is a reference line, not a series colour.
export const METHOD_COLOR = {
  pointer: 'var(--series-1)',
  markov: 'var(--series-2)',
  popularity: 'var(--series-3)',
  bkt_gain: 'var(--series-4)',
  actual: 'var(--reference)',
};

export const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
