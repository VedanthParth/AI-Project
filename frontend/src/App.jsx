import { useCallback, useEffect, useRef, useState } from 'react';
import { Route } from 'lucide-react';
import { api } from './api';
import StudentList from './components/StudentList';
import StudentHeader from './components/StudentHeader';
import HistoryCard from './components/HistoryCard';
import PathCard from './components/PathCard';
import CandidateTable from './components/CandidateTable';
import MethodComparison from './components/MethodComparison';
import SimulationChart from './components/SimulationChart';
import PrereqGraph from './components/PrereqGraph';
import TutorPanel from './components/TutorPanel';
import ModelCard from './components/ModelCard';

function readUrl() {
  const params = new URLSearchParams(window.location.search);
  const num = (key) => (params.has(key) ? Number(params.get(key)) : null);
  return { student: num('student'), example: num('moment') };
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [students, setStudents] = useState([]);
  const [studentId, setStudentId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [exampleId, setExampleId] = useState(null);
  const [example, setExample] = useState(null);
  const [simulation, setSimulation] = useState(null);
  const [graph, setGraph] = useState(null);
  const [errors, setErrors] = useState({});
  const initial = useRef(readUrl());
  const current = useRef(null);

  const selectExample = useCallback((id, sid) => {
    current.current = id;
    const stillCurrent = () => current.current === id;
    setExampleId(id);
    setExample(null);
    setSimulation(null);
    setGraph(null);
    setErrors({});
    const url = new URL(window.location.href);
    url.searchParams.set('student', sid);
    url.searchParams.set('moment', id);
    window.history.replaceState(null, '', url);
    const fail = (key) => (err) => stillCurrent() && setErrors((e) => ({ ...e, [key]: err.message }));
    api.example(id).then((x) => stillCurrent() && setExample(x)).catch(fail('example'));
    api.simulate(id).then((x) => stillCurrent() && setSimulation(x)).catch(fail('simulation'));
    api.graph(id).then((x) => stillCurrent() && setGraph(x)).catch(fail('graph'));
  }, []);

  const selectStudent = useCallback((sid) => {
    setStudentId(sid);
    setDetail(null);
    api.student(sid).then((d) => {
      setDetail(d);
      const wanted = d.examples.find((e) => e.id === initial.current.example);
      initial.current.example = null;
      const chosen = wanted ?? d.examples[Math.floor(d.examples.length / 2)];
      if (chosen) selectExample(chosen.id, sid);
    }).catch((err) => setErrors({ load: err.message }));
  }, [selectExample]);

  useEffect(() => {
    Promise.all([api.info(), api.students()])
      .then(([i, s]) => {
        setInfo(i);
        setStudents(s);
        const wanted = s.find((x) => x.id === initial.current.student);
        const first = wanted ?? [...s].sort((a, b) => b.examples - a.examples)[0];
        if (first) selectStudent(first.id);
      })
      .catch((err) => setErrors({ load: err.message }));
  }, [selectStudent]);

  const summary = students.find((s) => s.id === studentId);

  return (
    <div className="max-w-[1400px] mx-auto p-4 flex flex-col gap-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'var(--accent)', color: '#fff' }}>
            <Route size={20} aria-hidden />
          </div>
          <div>
            <div className="text-lg font-semibold leading-tight">PathTrace</div>
            <div className="muted text-xs">Learning-path recommendation with a pointer network, on real Junyi Academy students</div>
          </div>
        </div>
        {info && (
          <div className="flex flex-wrap gap-2">
            <span className="chip num">Test NDCG@3 {info.metrics.pointer.ndcg.mean.toFixed(3)}</span>
            {info.metrics.markov && <span className="chip num">Markov {info.metrics.markov.ndcg.mean.toFixed(3)}</span>}
            <span className="chip num">{info.num_concepts.toLocaleString()} concepts</span>
          </div>
        )}
      </header>

      {errors.load && (
        <div className="card bad">Couldn't reach the PathTrace API ({errors.load}). Is <code>python backend/server.py</code> running?</div>
      )}

      <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)] items-start">
        <aside>
          {students.length > 0
            ? <StudentList students={students} selected={studentId} onSelect={selectStudent} />
            : !errors.load && <div className="skeleton" style={{ height: 400 }} />}
        </aside>

        <main className="flex flex-col gap-4 min-w-0">
          {summary && <StudentHeader summary={summary} detail={detail} exampleId={exampleId} onSelectExample={(id) => selectExample(id, studentId)} />}
          {errors.example && <div className="card bad">Couldn't load this moment: {errors.example}</div>}
          {!example && !errors.example && studentId !== null && (
            <div className="grid gap-4 xl:grid-cols-2">
              <div className="skeleton" style={{ height: 260 }} />
              <div className="skeleton" style={{ height: 260 }} />
            </div>
          )}
          {example && (
            <>
              <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)] items-start">
                <PathCard example={example} />
                <HistoryCard example={example} />
              </div>
              <MethodComparison example={example} simulation={simulation} />
              <div className="grid gap-4 xl:grid-cols-2 items-start">
                <SimulationChart example={example} simulation={simulation} error={errors.simulation} />
                <TutorPanel key={example.id} example={example} />
              </div>
              <CandidateTable example={example} />
              <PrereqGraph example={example} graph={graph} error={errors.graph} trainStudents={info?.train_students} />
            </>
          )}
          {info && <ModelCard info={info} />}
        </main>
      </div>
    </div>
  );
}
