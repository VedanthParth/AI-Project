import { AlertTriangle, ArrowRight, Check, RotateCcw, X } from 'lucide-react';
import { PLAN_COLOR, PLAN_ORDER, fmtNum, pct } from '../../api';

const MARK_LABEL = { good: 'confident', bad: 'needs help', inferred: 'inferred' };

function Stage({ value, label, note }) {
  return (
    <div className="rounded-lg px-3 py-2 min-w-[120px]" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }}>
      <div className="text-lg font-semibold num">{value}</div>
      <div className="text-xs">{label}</div>
      {note && <div className="muted text-[11px]">{note}</div>}
    </div>
  );
}

export function PipelineStrip({ plan }) {
  const p = plan.pipeline;
  const stages = [
    { value: fmtNum(p.catalogue), label: 'exercises in Junyi' },
    { value: fmtNum(p.stage_concepts), label: 'for this class group' },
    { value: fmtNum(p.eligible), label: 'not done yet', note: `${p.history_concepts} known from the answers` },
    { value: p.shortlist_size, label: 'shortlisted', note: 'Markov: what usually follows the latest exercises' },
    { value: 3, label: 'ordered into a path', note: `pointer network (${p.models.pointer})` },
  ];
  return (
    <section className="card flex flex-col gap-2" aria-label="How the path is chosen">
      <h2 className="card-title">2. How the path is chosen</h2>
      <div className="flex flex-wrap items-center gap-2">
        {stages.map((s, i) => (
          <div key={s.label} className="flex items-center gap-2">
            {i > 0 && <ArrowRight size={16} className="muted" aria-hidden />}
            <Stage {...s} />
          </div>
        ))}
      </div>
    </section>
  );
}

function Attention({ step, shortlist, chosen }) {
  const top = step.map((p, slot) => ({ p, slot })).sort((a, b) => b.p - a.p).slice(0, 5);
  return (
    <ul className="m-0 p-0 list-none flex flex-col gap-1" aria-label="Model attention at this step">
      {top.map(({ p, slot }) => (
        <li key={slot} className="grid grid-cols-[minmax(0,1fr)_90px_40px] items-center gap-2 text-[11px]">
          <span className={slot === chosen ? 'font-semibold truncate' : 'muted truncate'} title={shortlist[slot].name}>
            {shortlist[slot].name}
          </span>
          <span className="bar-track"><span className="bar-fill" style={{ width: pct(p), background: slot === chosen ? 'var(--accent)' : 'var(--muted)' }} /></span>
          <span className="num text-right">{pct(p)}</span>
        </li>
      ))}
    </ul>
  );
}

function changes(plan, prev) {
  if (!prev) return null;
  const now = plan.paths.pointer.concepts.map((c) => c.id);
  const before = prev.paths.pointer.concepts.map((c) => c.id);
  if (now.join() === before.join()) return { same: true };
  const names = Object.fromEntries([...plan.paths.pointer.concepts, ...prev.paths.pointer.concepts].map((c) => [c.id, c.name]));
  return { added: now.filter((c) => !before.includes(c)).map((c) => names[c]), dropped: before.filter((c) => !now.includes(c)).map((c) => names[c]) };
}

export function PathSteps({ plan, prev, lastAction, onPractice }) {
  const path = plan.paths.pointer;
  const diff = changes(plan, prev);
  return (
    <section className="card flex flex-col gap-3" aria-label="Recommended path">
      <div>
        <h2 className="card-title">3. Recommended path</h2>
        <p className="card-sub">
          At each step the pointer network spreads its attention over the shortlist (top 5 shown), picks the highest, and
          moves on without it. Record how the student does and the path is recomputed.
        </p>
      </div>
      {diff && lastAction && (
        <div className="rounded-lg px-3 py-2 text-xs" style={{ background: 'var(--accent-soft)' }} role="status">
          <span className="font-medium">{lastAction}.</span>{' '}
          {diff.same ? 'The path stays the same.' : (
            <>
              {diff.added.length > 0 && <>Now includes <b>{diff.added.join(', ')}</b>. </>}
              {diff.dropped.length > 0 && <>Dropped {diff.dropped.join(', ')}.</>}
            </>
          )}
        </div>
      )}
      <ol className="m-0 p-0 list-none grid gap-3 lg:grid-cols-3">
        {path.concepts.map((c, step) => {
          const cand = plan.shortlist[c.slot];
          return (
            <li key={c.id} className="rounded-lg p-3 flex flex-col gap-2" style={{ border: '1px solid var(--border)', background: 'var(--surface-2)' }}>
              <div className="flex items-start gap-2">
                <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                  style={{ background: 'var(--accent)', color: '#fff' }}>{step + 1}</span>
                <div className="min-w-0">
                  <div className="font-medium leading-snug">{c.name}</div>
                  <div className="muted text-xs">{c.unit}</div>
                </div>
              </div>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-0.5 m-0 text-xs">
                <dt className="muted">BKT mastery now</dt><dd className="m-0 num">{pct(cand.mastery)}</dd>
                <dt className="muted">Expected gain</dt><dd className="m-0 num">{cand.bkt_gain.toFixed(3)}</dd>
              </dl>
              {c.unmet_prereqs.length > 0 ? (
                <div className="flex items-start gap-1.5 text-xs">
                  <AlertTriangle size={14} style={{ color: 'var(--series-4)' }} className="shrink-0 mt-0.5" aria-hidden />
                  <span>Inferred prerequisite not done yet: {c.unmet_prereqs.map((p) => p.name).join(', ')}</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-xs"><Check size={14} className="good" aria-hidden /> Prerequisites covered</div>
              )}
              <Attention step={path.steps[step]} shortlist={plan.shortlist} chosen={c.slot} />
              <div className="flex gap-2 mt-auto pt-1">
                <button type="button" className="btn flex-1 justify-center" onClick={() => onPractice(c, true)}>
                  <Check size={14} className="good" aria-hidden /> Got it
                </button>
                <button type="button" className="btn flex-1 justify-center" onClick={() => onPractice(c, false)}>
                  <X size={14} className="bad" aria-hidden /> Struggled
                </button>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export function KnownPanel({ plan, onResetPractice, practiced }) {
  return (
    <section className="card flex flex-col gap-3" aria-label="What the system knows">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="card-title">What the system knows</h2>
          <p className="card-sub">The answers become a short attempt history; BKT replays it into mastery estimates.</p>
        </div>
        {practiced > 0 && (
          <button type="button" className="btn shrink-0" onClick={onResetPractice}><RotateCcw size={14} aria-hidden /> Clear practice</button>
        )}
      </div>
      {plan.units.length > 0 && (
        <ul className="m-0 p-0 list-none flex flex-col gap-1">
          {plan.units.map((u) => (
            <li key={u.id} className="grid grid-cols-[minmax(0,1fr)_100px_44px] items-center gap-2 text-xs">
              <span className="truncate" title={u.name}>{u.name} <span className="muted">({MARK_LABEL[u.mark] ?? u.mark})</span></span>
              <span className="bar-track"><span className="bar-fill" style={{ width: pct(u.mastery) }} /></span>
              <span className="num text-right">{pct(u.mastery)}</span>
            </li>
          ))}
        </ul>
      )}
      {plan.review.length > 0 && (
        <div className="rounded-lg p-2 text-xs" style={{ border: '1px solid var(--border)' }}>
          <div className="font-medium mb-1">Review first <span className="muted font-normal">(rule: mastery below {pct(plan.review_below)}; the models only pick new exercises)</span></div>
          <ul className="m-0 pl-4">
            {plan.review.map((r) => <li key={r.id}>{r.name} <span className="muted num">{pct(r.mastery)}</span></li>)}
          </ul>
        </div>
      )}
      <details className="text-xs">
        <summary className="cursor-pointer">History built from the answers ({plan.history.length} exercises)</summary>
        <ul className="m-0 mt-1 pl-4 flex flex-col gap-0.5">
          {plan.history.map((h) => (
            <li key={h.id}>{h.name} <span className="muted">· {h.correct}/{h.attempts} right · {h.source}</span></li>
          ))}
        </ul>
      </details>
    </section>
  );
}

export function CompareTable({ plan }) {
  const sim = plan.simulation.paths;
  const about = {
    pointer: 'Trained to follow what real students did next',
    rl: 'The same network, fine-tuned for simulated learning gain',
    markov: 'What usually follows the latest exercises',
  };
  return (
    <section className="card flex flex-col gap-3" aria-label="Compare ways of choosing">
      <div>
        <h2 className="card-title">4. Three ways to choose from the same shortlist</h2>
        <p className="card-sub">
          Expected gain is BKT's predicted mastery increase; skips count path exercises started before an inferred
          prerequisite; DKT gain simulates the student practising the path ({plan.simulation.attempts_per_concept} attempts
          each, {plan.simulation.rollouts} runs).
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr><th scope="col">Method</th><th scope="col">Path</th><th scope="col" className="text-right">Expected gain</th>
              <th scope="col" className="text-right">Skips</th><th scope="col" className="text-right">DKT gain</th></tr>
          </thead>
          <tbody>
            {PLAN_ORDER.filter((m) => plan.paths[m]).map((m) => {
              const p = plan.paths[m];
              return (
                <tr key={m} style={m === 'pointer' ? { background: 'var(--accent-soft)' } : undefined}>
                  <td style={{ minWidth: 160 }}>
                    <div className="flex items-center gap-2 font-medium">
                      <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: PLAN_COLOR[m] }} aria-hidden />
                      {p.label}
                    </div>
                    <div className="muted text-xs">{about[m]}</div>
                  </td>
                  <td style={{ minWidth: 240 }}><ol className="m-0 pl-4 text-xs">{p.concepts.map((c) => <li key={c.id}>{c.name}</li>)}</ol></td>
                  <td className="num text-right">{p.bkt_gain.toFixed(3)}</td>
                  <td className="num text-right">{p.violations}</td>
                  <td className="num text-right">{sim[m] ? sim[m].gain.toFixed(3) : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function ShortlistTable({ plan }) {
  const inPath = new Set(plan.paths.pointer.concepts.map((c) => c.id));
  return (
    <details className="card">
      <summary className="cursor-pointer card-title">The shortlist ({plan.shortlist.length} exercises)</summary>
      <div className="overflow-x-auto mt-2">
        <table className="table">
          <thead>
            <tr><th scope="col" className="text-right">#</th><th scope="col">Exercise</th><th scope="col">Chapter</th>
              <th scope="col" className="text-right">Markov</th><th scope="col" className="text-right">Mastery</th>
              <th scope="col" className="text-right">Exp. gain</th></tr>
          </thead>
          <tbody>
            {plan.shortlist.map((s) => (
              <tr key={s.id} style={inPath.has(s.id) ? { background: 'var(--accent-soft)' } : undefined}>
                <td className="num text-right">{s.rank}</td>
                <td>{s.name}{s.weak_unit && <span className="chip ml-1">weak chapter</span>}</td>
                <td className="muted text-xs">{s.unit}</td>
                <td className="num text-right">{s.markov.toFixed(4)}</td>
                <td className="num text-right">{pct(s.mastery)}</td>
                <td className="num text-right">{s.bkt_gain.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
