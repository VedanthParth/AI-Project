import { useEffect, useState } from 'react';
import { api, fmtNum } from '../api';

const METRICS = [['ndcg', 'NDCG@3'], ['hit_rate', 'Hit rate'], ['first_step', 'First pick right']];

const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(d));

function MeanCI({ m }) {
  if (!m) return '—';
  return (
    <span className="num">
      {fmt(m.mean)}
      {m.ci_low !== undefined && <span className="muted text-[11px]"> [{fmt(m.ci_low)}, {fmt(m.ci_high)}]</span>}
    </span>
  );
}

function Section({ title, sub, children }) {
  return (
    <section className="card flex flex-col gap-3">
      <div>
        <h2 className="card-title">{title}</h2>
        {sub && <p className="card-sub">{sub}</p>}
      </div>
      <div className="overflow-x-auto">{children}</div>
    </section>
  );
}

// {method: {metric: {mean, ci_low, ci_high}}}
function MethodTable({ methods, highlight = (name) => name.startsWith('pointer') }) {
  return (
    <table className="table">
      <thead>
        <tr><th scope="col">Method</th>{METRICS.map(([, label]) => <th key={label} scope="col" className="text-right">{label}</th>)}</tr>
      </thead>
      <tbody>
        {Object.entries(methods).map(([name, m]) => (
          <tr key={name} style={highlight(name) ? { background: 'var(--accent-soft)' } : undefined}>
            <td>{name}</td>
            {METRICS.map(([key]) => <td key={key} className="text-right"><MeanCI m={m[key]} /></td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MainTable({ main }) {
  const ms = (v, d = 3) => (Array.isArray(v) ? `${fmt(v[0], d)}${v[1] !== null && v[1] !== undefined ? ` ± ${fmt(v[1], d)}` : ''}` : '—');
  return (
    <table className="table">
      <thead>
        <tr>
          <th scope="col">Method</th><th scope="col" className="text-right">Seeds</th>
          <th scope="col" className="text-right">NDCG@3</th><th scope="col" className="text-right">First pick right</th>
          <th scope="col" className="text-right">BKT gain / path</th><th scope="col" className="text-right">Skips / path</th>
          <th scope="col" className="text-right">DKT gain</th><th scope="col">vs strongest baseline</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(main.rows).map(([name, r]) => (
          <tr key={name} style={name === 'pointer' || name === 'rl' ? { background: 'var(--accent-soft)' } : undefined}>
            <td>{r.label}</td>
            <td className="num text-right">{r.seeds}</td>
            <td className="num text-right">{ms(r.ndcg)}</td>
            <td className="num text-right">{ms(r.first_step)}</td>
            <td className="num text-right">{ms(r.bkt_gain)}</td>
            <td className="num text-right">{ms(r.violations, 2)}</td>
            <td className="num text-right">{ms(r.dkt_gain, 4)}</td>
            <td className="num text-xs">
              {r.vs_strongest
                ? `${r.vs_strongest.mean >= 0 ? '+' : ''}${fmt(r.vs_strongest.mean)} [${fmt(r.vs_strongest.ci_low)}, ${fmt(r.vs_strongest.ci_high)}]`
                : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Results() {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.results().then(setResults).catch((err) => setError(err.message)); }, []);

  if (error) return <div className="card bad">Couldn't load the results ({error}).</div>;
  if (!results) return <div className="skeleton" style={{ height: 300 }} />;
  if (Object.keys(results).length === 0) {
    return <div className="card muted">This bundle has no results yet. Re-export it with <code>--results</code> after the runs finish.</div>;
  }

  const catalogue = Object.entries(results).filter(([k]) => k.startsWith('catalogue_test_'));
  const onboarding = Object.entries(results).filter(([k]) => k.startsWith('onboarding'));
  const folds = Object.entries(results).filter(([k]) => k.startsWith('walkforward_')).sort();

  return (
    <div className="flex flex-col gap-4">
      {results.main && (
        <Section title="Held-out students, 20 candidates each"
          sub={`The main experiment: ${fmtNum(results.main.examples)} test moments from students no model was trained on. Each moment's 20 candidates contain the 3 real next exercises, so this measures ordering, not finding them. Mean ± std over seeds; the last column is the paired difference in NDCG@3 with a 95% interval from resampling whole students.`}>
          <MainTable main={results.main} />
        </Section>
      )}
      {catalogue.map(([key, r]) => (
        <Section key={key} title={`Whole catalogue, shortlist of ${r.shortlist_size}`}
          sub={`No list that holds the answer: the Markov shortlist is drawn from all ${fmtNum(Math.round(r.eligible_concepts))} unseen exercises and holds ${(r.shortlist.target_recall * 100).toFixed(0)}% of the real next ones; every method is scored against what the student really did (${fmtNum(r.examples)} test moments).`}>
          <MethodTable methods={r.methods} />
        </Section>
      ))}
      {onboarding.map(([key, r]) => (
        <Section key={key} title={`New-student questionnaire (simulated${r.settings.current ? `, naming the last ${r.settings.current} exercise${r.settings.current > 1 ? 's' : ''}` : ''}${r.settings.noise ? `, ${Math.round(r.settings.noise * 100)}% of answers flipped` : ''})`}
          sub={`Held-out students' real histories fill in the questionnaire (${r.questions_answered.toFixed(1)} units rated on average); recommendations from those answers alone are scored against what each student did next, next to the same methods given the full history.`}>
          <MethodTable methods={r.methods} highlight={(name) => name.includes('questionnaire')} />
        </Section>
      ))}
      {folds.map(([key, r]) => (
        <Section key={key} title={`Walk-forward fold from ${r.origin}`}
          sub={`Trained only on activity before ${r.origin} (minus 30 days for validation); tested on the following 60 days. New students had no activity before ${r.origin}.`}>
          {['all', 'new', 'continuing'].map((g) => (
            <div key={g} className="mb-3">
              <div className="text-sm font-medium mb-1">{g[0].toUpperCase() + g.slice(1)} students <span className="muted num">({fmtNum(r.examples[g])} moments)</span></div>
              <MethodTable methods={Object.fromEntries(Object.entries(r.methods).filter(([, m]) => m[g]).map(([n, m]) => [n, m[g]]))} />
            </div>
          ))}
        </Section>
      ))}
    </div>
  );
}
