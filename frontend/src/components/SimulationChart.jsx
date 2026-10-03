import { useRef, useState } from 'react';
import { METHOD_COLOR, METHOD_ORDER, pct } from '../api';

const W = 640;
const H = 260;
const M = { top: 12, right: 16, bottom: 40, left: 48 };

function niceDomain(values) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = Math.max((hi - lo) * 0.15, 0.005);
  return [Math.max(0, lo - pad), Math.min(1, hi + pad)];
}

function ticks([lo, hi], count = 4) {
  const step = (hi - lo) / count;
  return Array.from({ length: count + 1 }, (_, i) => lo + i * step);
}

export default function SimulationChart({ example, simulation, error }) {
  const [hover, setHover] = useState(null);
  const svgRef = useRef(null);

  if (error) return <section className="card"><p className="bad m-0">Simulation failed: {error}</p></section>;
  if (!simulation) return <section className="card"><div className="skeleton" style={{ height: 300 }} /></section>;

  const series = METHOD_ORDER.filter((m) => simulation.paths[m]).map((m) => ({
    name: m, label: example.paths[m].label, ...simulation.paths[m],
  }));
  const n = series[0].curve.length;
  const per = simulation.attempts_per_concept;
  const domain = niceDomain(series.flatMap((s) => s.curve));
  const x = (i) => M.left + (i / (n - 1)) * (W - M.left - M.right);
  const y = (v) => M.top + (1 - (v - domain[0]) / (domain[1] - domain[0])) * (H - M.top - M.bottom);

  const onMove = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((px - M.left) / (W - M.left - M.right)) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <section className="card flex flex-col gap-3" aria-label="Simulated practice">
      <div>
        <h2 className="card-title">Simulate following each path</h2>
        <p className="card-sub">
          A separate DKT model, never used in training, plays the student: {per} attempts on each path concept, answers
          drawn from its predictions, averaged over {simulation.rollouts} runs. The line is the predicted chance of
          answering the {example.candidates.length} candidates correctly.
        </p>
      </div>
      <ul className="m-0 p-0 list-none flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
        {series.map((s) => (
          <li key={s.name} className="flex items-center gap-1.5">
            <svg width="18" height="8" aria-hidden>
              <line x1="0" y1="4" x2="18" y2="4" stroke={METHOD_COLOR[s.name]} strokeWidth="2"
                strokeDasharray={s.name === 'actual' ? '4 3' : undefined} />
            </svg>
            <span>{s.label}</span>
            <span className="muted num">gain {s.gain.toFixed(3)}</span>
          </li>
        ))}
      </ul>
      <div className="relative">
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
          aria-label="Predicted success on the candidates after each simulated attempt, per method"
          onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {ticks(domain).map((t) => (
            <g key={t}>
              <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth="1" />
              <text x={M.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize="11" fill="var(--muted)">
                {pct(t)}
              </text>
            </g>
          ))}
          {Array.from({ length: Math.floor((n - 1) / per) }, (_, c) => (c + 1) * per).map((i) => (
            <g key={i}>
              <line x1={x(i)} x2={x(i)} y1={M.top} y2={H - M.bottom} stroke="var(--grid)" strokeDasharray="2 3" />
              <text x={x(i)} y={H - M.bottom + 16} textAnchor={i === n - 1 ? 'end' : 'middle'} fontSize="11" fill="var(--muted)">
                after concept {i / per}
              </text>
            </g>
          ))}
          <text x={x(0)} y={H - M.bottom + 16} textAnchor="start" fontSize="11" fill="var(--muted)">now</text>
          <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--axis)" />
          {series.map((s) => (
            <polyline key={s.name} fill="none" stroke={METHOD_COLOR[s.name]} strokeWidth="2" strokeLinejoin="round"
              strokeDasharray={s.name === 'actual' ? '5 4' : undefined}
              points={s.curve.map((v, i) => `${x(i)},${y(v)}`).join(' ')} />
          ))}
          {hover !== null && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} stroke="var(--axis)" />
              {series.map((s) => (
                <circle key={s.name} cx={x(hover)} cy={y(s.curve[hover])} r="4" fill={METHOD_COLOR[s.name]}
                  stroke="var(--surface)" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="tooltip" style={{ left: `${(x(hover) / W) * 100}%`, top: 0, transform: `translateX(${hover > n / 2 ? '-105%' : '5%'})` }}>
            <div className="font-medium mb-1">{hover === 0 ? 'Before practice' : `Attempt ${hover} (concept ${Math.ceil(hover / per)})`}</div>
            {[...series].sort((a, b) => b.curve[hover] - a.curve[hover]).map((s) => (
              <div key={s.name} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-2 h-2 rounded-full" style={{ background: METHOD_COLOR[s.name] }} />
                  {s.label}
                </span>
                <span className="num">{pct(s.curve[hover], 1)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <details className="text-xs">
        <summary className="cursor-pointer muted">Table view</summary>
        <table className="table mt-2">
          <thead>
            <tr><th>Method</th><th className="text-right">Before</th><th className="text-right">After</th><th className="text-right">Normalised gain</th></tr>
          </thead>
          <tbody>
            {series.map((s) => (
              <tr key={s.name}>
                <td>{s.label}</td>
                <td className="num text-right">{pct(s.before, 1)}</td>
                <td className="num text-right">{pct(s.after, 1)}</td>
                <td className="num text-right">{s.gain.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
