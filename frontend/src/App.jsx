import { useEffect, useState } from 'react';
import { Route } from 'lucide-react';
import { api } from './api';
import { APP } from './copy';
import Plan from './pages/Plan';
import Examples from './pages/Examples';
import HowItWorks from './pages/HowItWorks';

const PAGES = [
  ['plan', 'Plan my next topics'],
  ['examples', 'See it on real students'],
  ['how', 'How it works'],
];
// Older links used these names for the same pages.
const OLD_NAMES = { new: 'plan', students: 'examples', results: 'how' };

function pageFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const name = OLD_NAMES[params.get('tab')] ?? params.get('tab');
  if (PAGES.some(([id]) => id === name)) return name;
  return params.has('student') ? 'examples' : 'plan';
}

export default function App() {
  const [page, setPage] = useState(pageFromUrl);
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.info().then(setInfo).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    // Keep the open page in the address bar, so a reload or a shared link opens the same page.
    const url = new URL(window.location.href);
    url.searchParams.set('tab', page);
    window.history.replaceState(null, '', url);
  }, [page]);

  const open = (id) => {
    setPage(id);
    window.scrollTo({ top: 0 });
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-20" style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <div className="mx-auto flex flex-wrap items-center justify-between gap-3 px-4 py-3" style={{ maxWidth: 1280 }}>
          <button type="button" className="flex items-center gap-3 text-left" onClick={() => open('plan')}>
            <span className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0" style={{ background: 'var(--accent)', color: '#fff' }}>
              <Route size={20} aria-hidden />
            </span>
            <span>
              <span className="block font-semibold leading-tight">{APP.name}</span>
              <span className="block muted text-xs">{APP.tagline}</span>
            </span>
          </button>
          <nav className="flex flex-wrap gap-1" aria-label="Pages">
            {PAGES.map(([id, label]) => (
              <button key={id} type="button" className="px-3 py-2 rounded-lg text-sm" aria-current={page === id ? 'page' : undefined}
                onClick={() => open(id)}
                style={page === id ? { background: 'var(--accent-soft)', color: 'var(--accent)', fontWeight: 600 } : { color: 'var(--ink-2)' }}>
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {error && (
        <div className="page" style={{ paddingBottom: 0 }}>
          <div className="card bad">Couldn't reach the app's server ({error}). Is <code>python backend/server.py</code> running?</div>
        </div>
      )}

      <div className="flex-1">
        {page === 'plan' && <Plan tutorMode={info?.tutor} />}
        {page === 'examples' && <Examples tutorMode={info?.tutor} />}
        {page === 'how' && <HowItWorks />}
      </div>

      {info && (
        <footer className="muted text-xs" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="mx-auto px-4 py-6" style={{ maxWidth: 1280 }}>
            Learned from the <a href={info.dataset.url} style={{ color: 'var(--accent)' }}>{info.dataset.name}</a>, licensed{' '}
            <a href={info.dataset.license_url} style={{ color: 'var(--accent)' }}>{info.dataset.license}</a>. Exercise names are machine-translated.
          </div>
        </footer>
      )}
    </div>
  );
}
