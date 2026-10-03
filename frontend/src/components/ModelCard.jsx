const ROWS = [
  ['pointer', 'Pointer network'],
  ['gru_next_item', 'GRU next-item'],
  ['markov', 'Markov'],
  ['popularity', 'Popularity'],
  ['random', 'Random'],
];

export default function ModelCard({ info }) {
  const m = info.metrics;
  return (
    <footer className="card flex flex-col gap-3 text-sm" aria-label="About the model">
      <div>
        <h2 className="card-title">About this model</h2>
        <p className="card-sub">
          Checkpoint <code>{info.source.checkpoint}</code> trained on {info.train_students.toLocaleString()} students
          from <code>{info.source.data}</code>. Scores on the full {info.source.split} split, with 95% bootstrap intervals.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Method</th>
              <th scope="col" className="text-right">NDCG@3</th>
              <th scope="col" className="text-right">Precision@3</th>
              <th scope="col" className="text-right">First pick right</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.filter(([key]) => m[key]).map(([key, label]) => (
              <tr key={key} style={key === 'pointer' ? { background: 'var(--accent-soft)' } : undefined}>
                <td>{label}</td>
                {['ndcg', 'precision', 'first_step'].map((metric) => (
                  <td key={metric} className="num text-right">
                    {m[key][metric].mean.toFixed(3)}
                    {m[key][metric].ci_low !== undefined && (
                      <span className="muted text-xs"> [{m[key][metric].ci_low.toFixed(3)}, {m[key][metric].ci_high.toFixed(3)}]</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted text-xs m-0">
        {m.markov && m.markov.first_step.mean > m.pointer.first_step.mean
          ? "Markov still picks the first concept best: a student's next concept usually follows the last few in Junyi's curriculum order. "
          : ''}
        DKT simulator test AUC: {info.dkt_auc ? info.dkt_auc.toFixed(3) : 'n/a'}.
      </p>
      <p className="muted text-xs m-0">
        Data: <a href={info.dataset.url} style={{ color: 'var(--accent)' }}>{info.dataset.name}</a>, licensed{' '}
        <a href={info.dataset.license_url} style={{ color: 'var(--accent)' }}>{info.dataset.license}</a>. Student
        labels are replaced with numbers. Concept names are machine-translated from Chinese. Mastery is a BKT estimate,
        not a test score.
      </p>
    </footer>
  );
}
