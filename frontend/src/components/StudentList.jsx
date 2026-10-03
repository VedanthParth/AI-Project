import { useEffect, useMemo, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { pct } from '../api';

const SORTS = {
  attempts: { label: 'Attempts', fn: (a, b) => b.attempts - a.attempts },
  accuracy: { label: 'Accuracy', fn: (a, b) => b.accuracy - a.accuracy },
  id: { label: 'Number', fn: (a, b) => a.id - b.id },
};

export default function StudentList({ students, selected, onSelect }) {
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('attempts');
  const [stage, setStage] = useState('');

  const listRef = useRef(null);
  useEffect(() => {
    listRef.current?.querySelector('[aria-current="true"]')?.scrollIntoView({ block: 'nearest' });
  }, [selected]);

  const stages = useMemo(() => [...new Set(students.map((s) => s.stage).filter(Boolean))].sort(), [students]);
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return students
      .filter((s) => (!q || s.label.toLowerCase().includes(q)) && (!stage || s.stage === stage))
      .sort(SORTS[sort].fn);
  }, [students, query, sort, stage]);

  return (
    <div className="card flex flex-col gap-3 lg:sticky lg:top-4" style={{ maxHeight: 'calc(100vh - 32px)' }}>
      <div>
        <h2 className="card-title">Held-out students</h2>
        <p className="card-sub">{students.length} test-split students no model was trained on</p>
      </div>
      <label className="relative block">
        <span className="sr-only">Search students</span>
        <Search size={14} className="absolute left-2.5 top-2.5 muted" aria-hidden />
        <input className="input" style={{ paddingLeft: 28 }} placeholder="Search, e.g. student-042" value={query}
          onChange={(e) => setQuery(e.target.value)} />
      </label>
      <div className="flex gap-2">
        <select className="input" value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort students by">
          {Object.entries(SORTS).map(([key, s]) => <option key={key} value={key}>{s.label}</option>)}
        </select>
        <select className="input" value={stage} onChange={(e) => setStage(e.target.value)} aria-label="Filter by stage">
          <option value="">All stages</option>
          {stages.map((s) => <option key={s} value={s}>{s[0].toUpperCase() + s.slice(1)}</option>)}
        </select>
      </div>
      <div ref={listRef} className="overflow-y-auto -mx-2 px-2 max-h-72 lg:max-h-none" role="list">
        {shown.map((s) => (
          <button key={s.id} type="button" role="listitem" className="list-item" aria-current={s.id === selected}
            onClick={() => onSelect(s.id)}>
            <div className="flex items-center justify-between gap-2">
              <span className="font-medium">{s.label}</span>
              <span className="muted num text-xs">{s.examples} moments</span>
            </div>
            <div className="muted text-xs num">
              {s.attempts.toLocaleString()} attempts · {s.concepts} concepts · {pct(s.accuracy)} correct
            </div>
          </button>
        ))}
        {!shown.length && <p className="muted text-sm px-2">No students match.</p>}
      </div>
    </div>
  );
}
