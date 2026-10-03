import { pct } from '../api';

export default function HistoryCard({ example }) {
  const h = example.history;
  return (
    <section className="card flex flex-col gap-3" aria-label="History before this moment">
      <div>
        <h2 className="card-title">Before this moment</h2>
        <p className="card-sub num">
          {example.date} · {h.attempts.toLocaleString()} attempts on {h.concepts} concepts · {pct(h.accuracy)} correct
        </p>
      </div>
      <div>
        <div className="flex justify-between muted text-xs mb-1">
          <span>Most recently practised</span>
          <span>BKT mastery estimate</span>
        </div>
        <ul className="m-0 p-0 list-none flex flex-col gap-2">
          {h.recent.map((r) => (
            <li key={r.id} className="grid gap-3 items-center" style={{ gridTemplateColumns: 'minmax(0,1fr) 120px' }}>
              <div className="min-w-0">
                <div className="leading-snug" title={r.name} style={{
                  display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
                }}>{r.name}</div>
                <div className="muted text-xs num">{r.correct}/{r.attempts} correct</div>
              </div>
              <div className="flex items-center gap-2">
                <div className="bar-track flex-1" aria-hidden>
                  <div className="bar-fill" style={{ width: pct(r.mastery) }} />
                </div>
                <span className="num text-xs w-9 text-right">{pct(r.mastery)}</span>
              </div>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
