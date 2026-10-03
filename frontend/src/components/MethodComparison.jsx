import { CheckCircle2, XCircle } from 'lucide-react';
import { METHOD_COLOR, METHOD_ORDER } from '../api';

const DESCRIPTION = {
  pointer: 'Our model',
  markov: 'What usually follows the last few concepts',
  popularity: 'Most-practised concepts overall',
  bkt_gain: 'Largest expected mastery gain under BKT',
  actual: 'Ground truth',
};

export default function MethodComparison({ example, simulation }) {
  return (
    <section className="card flex flex-col gap-3" aria-label="Compare methods">
      <div>
        <h2 className="card-title">Compare with the baselines</h2>
        <p className="card-sub">
          Same moment, same candidates. Prerequisite violations count path concepts started before an inferred
          prerequisite; DKT gain is the simulated improvement on all candidates after practising the path.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col">Path</th>
              <th scope="col" className="text-right">Matches</th>
              <th scope="col" className="text-center">First pick right</th>
              <th scope="col" className="text-right">Violations</th>
              <th scope="col" className="text-right">DKT gain</th>
            </tr>
          </thead>
          <tbody>
            {METHOD_ORDER.map((name) => {
              const p = example.paths[name];
              const sim = simulation?.paths?.[name];
              return (
                <tr key={name} style={name === 'pointer' ? { background: 'var(--accent-soft)' } : undefined}>
                  <td style={{ minWidth: 150 }}>
                    <div className="flex items-center gap-2 font-medium">
                      <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ background: METHOD_COLOR[name] }} aria-hidden />
                      {p.label}
                    </div>
                    <div className="muted text-xs">{DESCRIPTION[name]}</div>
                  </td>
                  <td style={{ minWidth: 240 }}>
                    <ol className="m-0 pl-4 text-xs">
                      {p.concepts.map((c, i) => (
                        <li key={i}>
                          {c.name}
                          {name !== 'actual' && example.candidates[p.slots[i]].target_rank && (
                            <CheckCircle2 size={12} className="good inline ml-1 align-[-2px]" aria-label="the student did this" />
                          )}
                        </li>
                      ))}
                    </ol>
                  </td>
                  <td className="num text-right">{p.hits}/3</td>
                  <td className="text-center">
                    {p.first_step
                      ? <CheckCircle2 size={16} className="good inline" aria-label="yes" />
                      : <XCircle size={16} className="bad inline" aria-label="no" />}
                  </td>
                  <td className="num text-right">{p.violations}</td>
                  <td className="num text-right">{sim ? sim.gain.toFixed(3) : '…'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
