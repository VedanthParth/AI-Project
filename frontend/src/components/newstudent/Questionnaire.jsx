import { useMemo, useState } from 'react';
import { Plus, Sparkles, X } from 'lucide-react';
import { STAGE_LABEL } from '../../api';

const RATING_LABEL = { strong: 'Strong', okay: 'Average', weak: 'Weak' };

function Segmented({ options, value, onChange, labels, size = 'sm' }) {
  return (
    <div className="inline-flex rounded-md overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      {options.map((opt) => (
        <button key={opt} type="button" onClick={() => onChange(value === opt ? null : opt)}
          className={size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3 py-1.5 text-sm'}
          aria-pressed={value === opt}
          style={value === opt ? { background: 'var(--accent)', color: '#fff' } : { background: 'var(--surface)' }}>
          {labels[opt]}
        </button>
      ))}
    </div>
  );
}

function UnitMarks({ units, marks, onMark }) {
  return (
    <ul className="m-0 p-0 list-none flex flex-col gap-1 mt-2 ml-1">
      {units.map((u) => (
        <li key={u.id} className="flex items-center justify-between gap-2 text-xs">
          <span className="ink-2">{u.name}</span>
          <Segmented options={['good', 'bad']} value={marks[u.id] ?? null} labels={{ good: 'Confident', bad: 'Needs help' }}
            onChange={(v) => onMark(u.id, v)} />
        </li>
      ))}
    </ul>
  );
}

function LastFinished({ units, current, onChange, max, names }) {
  const [unitId, setUnitId] = useState('');
  const [exerciseId, setExerciseId] = useState('');
  const unit = units.find((u) => String(u.id) === unitId);
  const add = () => {
    const id = Number(exerciseId);
    if (id && !current.includes(id) && current.length < max) onChange([...current, id]);
    setExerciseId('');
  };
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-2">
        <select className="input text-sm" value={unitId} onChange={(e) => { setUnitId(e.target.value); setExerciseId(''); }}
          aria-label="Chapter of the exercise">
          <option value="">Chapter…</option>
          {units.map((u) => <option key={u.id} value={u.id}>{u.name}</option>)}
        </select>
        <select className="input text-sm min-w-0 flex-1" value={exerciseId} onChange={(e) => setExerciseId(e.target.value)}
          disabled={!unit} aria-label="Exercise">
          <option value="">Exercise (in syllabus order)…</option>
          {unit?.exercises.map((e, i) => <option key={e.id} value={e.id}>{i + 1}. {e.name}</option>)}
        </select>
        <button type="button" className="btn" onClick={add} disabled={!exerciseId || current.length >= max}>
          <Plus size={14} aria-hidden /> Add
        </button>
      </div>
      {current.length > 0 && (
        <ol className="m-0 p-0 list-none flex flex-wrap gap-1.5">
          {current.map((id, i) => (
            <li key={id} className="chip flex items-center gap-1">
              <span className="muted">{i + 1}.</span> {names[id]}
              <button type="button" aria-label={`Remove ${names[id]}`} onClick={() => onChange(current.filter((c) => c !== id))}>
                <X size={12} />
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default function Questionnaire({ questionnaire, answers, onChange, onExample }) {
  const { stage, strands, units, current } = answers;
  const stageStrands = questionnaire.strands.filter((s) => s.stage === stage);
  const stageUnits = useMemo(() => questionnaire.units.filter((u) => u.stage === stage), [questionnaire, stage]);
  const names = useMemo(
    () => Object.fromEntries(questionnaire.units.flatMap((u) => u.exercises.map((e) => [e.id, e.name]))),
    [questionnaire],
  );
  const set = (patch) => onChange({ ...answers, ...patch });

  return (
    <section className="card flex flex-col gap-4" aria-label="New student questionnaire">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="card-title">1. Tell us about the student</h2>
          <p className="card-sub">Three short questions. The path updates after every answer.</p>
        </div>
        <button type="button" className="btn shrink-0" onClick={onExample}><Sparkles size={14} aria-hidden /> Example</button>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-sm font-medium">Class</span>
        <Segmented options={questionnaire.stages} value={stage} labels={STAGE_LABEL} size="md"
          onChange={(v) => v && onChange({ stage: v, strands: {}, units: {}, current: [], practice: [] })} />
      </div>

      <div>
        <div className="text-sm font-medium">How is the student doing in each area of maths?</div>
        <p className="muted text-xs m-0 mb-2">Strong marks all its chapters as known. Average or weak opens a checklist of its chapters.</p>
        <ul className="m-0 p-0 list-none flex flex-col gap-2">
          {stageStrands.map((s) => (
            <li key={s.id} className="rounded-lg p-2" style={{ border: '1px solid var(--border)' }}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm">{s.name}</span>
                <Segmented options={questionnaire.ratings} value={strands[s.id] ?? null} labels={RATING_LABEL}
                  onChange={(v) => {
                    const next = { ...strands };
                    if (v) next[s.id] = v; else delete next[s.id];
                    set({ strands: next });
                  }} />
              </div>
              {(strands[s.id] === 'okay' || strands[s.id] === 'weak') && (
                <UnitMarks units={stageUnits.filter((u) => u.strand === s.id)} marks={units}
                  onMark={(id, v) => {
                    const next = { ...units };
                    if (v) next[id] = v; else delete next[id];
                    set({ units: next });
                  }} />
              )}
            </li>
          ))}
        </ul>
      </div>

      <div>
        <div className="text-sm font-medium">Which exercises did the student finish last?</div>
        <p className="muted text-xs m-0 mb-2">
          Up to {questionnaire.max_current}, oldest first. This is the most useful answer: on held-out students, naming the
          last exercise recovers about half of what a full year of history gives; rating chapters alone recovers very little.
        </p>
        <LastFinished units={stageUnits} current={current} names={names} max={questionnaire.max_current}
          onChange={(c) => set({ current: c })} />
      </div>
    </section>
  );
}
