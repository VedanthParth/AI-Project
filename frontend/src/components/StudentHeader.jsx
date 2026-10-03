import { ChevronLeft, ChevronRight } from 'lucide-react';
import { pct } from '../api';

function Stat({ label, value }) {
  return (
    <div>
      <div className="muted text-xs">{label}</div>
      <div className="text-lg font-semibold num">{value}</div>
    </div>
  );
}

export default function StudentHeader({ summary, detail, exampleId, onSelectExample }) {
  const examples = detail?.examples ?? [];
  const index = examples.findIndex((e) => e.id === exampleId);
  const go = (delta) => {
    const next = examples[index + delta];
    if (next) onSelectExample(next.id);
  };

  return (
    <section className="card flex flex-col gap-4" aria-label="Student">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold m-0">{summary.label}</h1>
          <p className="card-sub">
            Active {summary.first_date} to {summary.last_date}
            {summary.stage ? ` · mostly ${summary.stage} concepts` : ''}
          </p>
        </div>
        <div className="flex gap-6">
          <Stat label="Attempts" value={summary.attempts.toLocaleString()} />
          <Stat label="Concepts" value={summary.concepts} />
          <Stat label="Correct" value={pct(summary.accuracy)} />
        </div>
      </div>
      <div>
        <div className="flex items-center justify-between gap-2 mb-2">
          <div>
            <div className="font-medium text-sm">Choose a moment in this student's history</div>
            <div className="muted text-xs">Each moment is a point where the student went on to start new concepts; the model only sees what came before it.</div>
          </div>
          <div className="flex gap-1 shrink-0">
            <button type="button" className="btn" onClick={() => go(-1)} disabled={index <= 0} aria-label="Previous moment">
              <ChevronLeft size={16} />
            </button>
            <button type="button" className="btn" onClick={() => go(1)} disabled={index < 0 || index >= examples.length - 1}
              aria-label="Next moment">
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Moments">
          {examples.map((e, i) => (
            <button key={e.id} type="button" role="tab" aria-selected={e.id === exampleId}
              className="btn num" onClick={() => onSelectExample(e.id)}
              style={e.id === exampleId ? { background: 'var(--accent)', borderColor: 'var(--accent)', color: '#fff' } : undefined}
              title={`After ${e.cut} attempts, ${e.date}`}>
              <span className="text-xs">#{i + 1}</span>
              <span className="text-xs opacity-80">{e.date}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
