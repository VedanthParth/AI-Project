import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import { pct } from '../api';

const ORDINAL = ['first', 'second', 'third'];

function Outcome({ cand, firstGroup }) {
  if (!cand.target_rank) {
    return (
      <div className="flex items-center gap-1.5 text-xs">
        <XCircle size={14} className="bad shrink-0" aria-hidden />
        <span>Not among the student's next 3 new concepts</span>
      </div>
    );
  }
  const first = cand.target_rank <= firstGroup;
  const text = first
    ? `The student started this next${firstGroup > 1 ? `, in the same session as ${firstGroup - 1} other new concept${firstGroup > 2 ? 's' : ''}` : ''}`
    : `The student started this ${ORDINAL[cand.target_rank - 1]} of their next 3 new concepts`;
  return (
    <div className="flex items-center gap-1.5 text-xs">
      <CheckCircle2 size={14} className="good shrink-0" aria-hidden />
      <span>{text}</span>
    </div>
  );
}

function listNames(items) {
  const names = items.map((p) => p.name);
  return names.length > 2 ? `${names.slice(0, 2).join(', ')} and ${names.length - 2} more` : names.join(' and ');
}

// Prerequisites the student hasn't started, split into ones an earlier step of the path covers and ones it doesn't.
function Prerequisites({ cand, earlier }) {
  const earlierIds = new Set(earlier.map((c) => c.id));
  const covered = cand.unmet_prereqs.filter((p) => earlierIds.has(p.id));
  const missing = cand.unmet_prereqs.filter((p) => !earlierIds.has(p.id));
  return (
    <>
      {covered.length > 0 && (
        <div className="flex items-start gap-1.5 text-xs">
          <CheckCircle2 size={14} className="good shrink-0 mt-0.5" aria-hidden />
          <span>Comes after its prerequisite {listNames(covered)}, earlier in the path</span>
        </div>
      )}
      {missing.length > 0 && (
        <div className="flex items-start gap-1.5 text-xs">
          <AlertTriangle size={14} style={{ color: 'var(--series-4)' }} className="shrink-0 mt-0.5" aria-hidden />
          <span>Inferred prerequisite not started: {listNames(missing)}</span>
        </div>
      )}
    </>
  );
}

export default function PathCard({ example }) {
  const path = example.paths.pointer;
  return (
    <section className="card flex flex-col gap-3" aria-label="Recommended path">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h2 className="card-title">Recommended path</h2>
          <p className="card-sub">
            The pointer network picks 3 concepts in order from {example.candidates.length} candidates. Attention is the
            share of the model's probability on that concept at that step.
          </p>
        </div>
        <span className="chip num">{path.hits}/3 match what the student did</span>
      </div>
      <ol className="m-0 p-0 list-none grid gap-3 md:grid-cols-3">
        {path.slots.map((slot, step) => {
          const cand = example.candidates[slot];
          const attention = example.pointer_steps[step][slot];
          return (
            <li key={slot} className="rounded-lg p-3 flex flex-col gap-2"
              style={{ border: '1px solid var(--border)', background: 'var(--surface-2)' }}>
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                  style={{ background: 'var(--accent)', color: '#fff' }}>{step + 1}</span>
                <span className="font-medium leading-snug">{cand.name}</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {cand.stage && <span className="chip">{cand.stage}</span>}
                {cand.level && <span className="chip">{cand.level}</span>}
              </div>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1 m-0 text-xs">
                <dt className="muted">Attention</dt>
                <dd className="m-0 num font-semibold">{pct(attention)}</dd>
                <dt className="muted">BKT mastery</dt>
                <dd className="m-0 num">{pct(cand.mastery)}</dd>
                <dt className="muted">Expected BKT gain</dt>
                <dd className="m-0 num">{cand.bkt_gain.toFixed(3)}</dd>
              </dl>
              <Prerequisites cand={cand} earlier={path.slots.slice(0, step).map((s) => example.candidates[s])} />
              <Outcome cand={cand} firstGroup={example.first_group} />
            </li>
          );
        })}
      </ol>
    </section>
  );
}
