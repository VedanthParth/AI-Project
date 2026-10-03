import { useState } from 'react';
import { METHOD_COLOR, pct } from '../api';

const SHORT = { pointer: 'Pointer', markov: 'Markov', popularity: 'Popular', bkt_gain: 'BKT' };
const RANK = ['1st', '2nd', '3rd'];

function HeatCell({ p, picked, masked }) {
  if (masked) {
    return <td className="num text-center muted" title="Already picked at an earlier step">–</td>;
  }
  // One-hue intensity scale; the value is always printed, so colour never carries it alone.
  const alpha = Math.min(0.9, Math.sqrt(p) * 0.9);
  return (
    <td className="num text-center text-xs" title={`${pct(p, 1)} of this step's attention`}
      style={{
        background: `rgba(var(--heat), ${alpha})`,
        color: alpha > 0.5 ? '#fff' : 'var(--ink)',
        outline: picked ? '2px solid var(--ink)' : undefined,
        outlineOffset: -2,
        fontWeight: picked ? 600 : 400,
      }}>
      {p >= 0.005 ? pct(p) : '<1%'}
    </td>
  );
}

export default function CandidateTable({ example }) {
  const [showAll, setShowAll] = useState(false);
  const pointer = example.paths.pointer.slots;
  const steps = example.pointer_steps;
  const order = [...example.candidates.keys()].sort((a, b) => {
    const ra = pointer.indexOf(a), rb = pointer.indexOf(b);
    if (ra !== -1 || rb !== -1) return (ra === -1 ? 99 : ra) - (rb === -1 ? 99 : rb);
    return steps[0][b] - steps[0][a];
  });
  const rows = showAll ? order : order.slice(0, 10);
  const pickedBy = (slot) =>
    ['pointer', 'markov', 'popularity', 'bkt_gain'].filter((m) => example.paths[m].slots.includes(slot));

  return (
    <section className="card flex flex-col gap-3" aria-label="Candidates and attention">
      <div>
        <h2 className="card-title">All {example.candidates.length} candidates and the model's attention</h2>
        <p className="card-sub">
          Darker cells got more of the decoder's attention at that step; outlined cells are its picks. Concepts picked
          earlier are masked at later steps.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Concept</th>
              <th scope="col" className="text-center">Step 1</th>
              <th scope="col" className="text-center">Step 2</th>
              <th scope="col" className="text-center">Step 3</th>
              <th scope="col">BKT mastery</th>
              <th scope="col">Picked by</th>
              <th scope="col">Student did</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((slot) => {
              const cand = example.candidates[slot];
              const pickStep = pointer.indexOf(slot);
              return (
                <tr key={slot}>
                  <td style={{ minWidth: 200 }}>
                    <div className="leading-snug">{cand.name}</div>
                    {cand.unmet_prereqs.length > 0 && (
                      <div className="muted text-xs">Unmet prerequisite: {cand.unmet_prereqs[0].name}</div>
                    )}
                  </td>
                  {steps.map((row, t) => (
                    <HeatCell key={t} p={row[slot]} picked={pickStep === t} masked={pickStep !== -1 && pickStep < t} />
                  ))}
                  <td className="num text-xs">{pct(cand.mastery)}</td>
                  <td>
                    <div className="flex flex-wrap gap-1">
                      {pickedBy(slot).map((m) => (
                        <span key={m} className="chip">
                          <span className="inline-block w-2 h-2 rounded-full" style={{ background: METHOD_COLOR[m] }} aria-hidden />
                          {SHORT[m]}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="text-xs">
                    {cand.target_rank ? <span className="font-medium">{RANK[cand.target_rank - 1]} of next 3</span> : <span className="muted">–</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {example.candidates.length > 10 && (
        <button type="button" className="btn self-start" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Show top 10' : `Show all ${example.candidates.length}`}
        </button>
      )}
    </section>
  );
}
