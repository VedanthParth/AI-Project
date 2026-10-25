// Pieces every chart shares: a legend whose labels are always visible (identity is never colour alone)
// and a table view, the accessible twin of each chart.

export function ChartLegend({ series }) {
  return (
    <ul className="m-0 p-0 list-none flex flex-wrap justify-center gap-x-5 gap-y-1.5 text-sm" aria-label="Legend">
      {series.map((s) => (
        <li key={s.key} className="flex items-center gap-2">
          <svg width="22" height="10" aria-hidden>
            <line x1="1" y1="5" x2="21" y2="5" stroke={s.color} strokeWidth="2.5" strokeDasharray={s.dashed ? '5 3' : undefined} strokeLinecap="round" />
          </svg>
          <span>{s.name}</span>
        </li>
      ))}
    </ul>
  );
}

export function TableView({ caption, columns, rows }) {
  return (
    <details className="text-sm">
      <summary className="cursor-pointer muted">Show as a table</summary>
      <div className="overflow-x-auto mt-2">
        <table className="table">
          {caption && <caption className="sr-only">{caption}</caption>}
          <thead>
            <tr><th scope="col" />{columns.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map(([label, ...cells]) => (
              <tr key={label}><th scope="row" className="font-normal text-left">{label}</th>{cells.map((c, i) => <td key={i}>{c}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
