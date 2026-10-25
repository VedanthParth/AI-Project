import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CheckCircle2, Clock, XCircle } from 'lucide-react';
import { api, fmtDate, PLAN_COLOR } from '../api';
import { EXAMPLE_AXES } from '../copy';
import Section from '../components/Section';
import StudentList from '../components/StudentList';
import StudentHeader from '../components/StudentHeader';
import LineChart from '../components/charts/LineChart';
import SpiderChart from '../components/charts/SpiderChart';
import { Compare, WhyTopic } from '../components/Explain';
import PrereqGraph from '../components/PrereqGraph';
import CandidateTable from '../components/CandidateTable';
import Tutor from '../components/Tutor';
import TechnicalDetails from '../components/TechnicalDetails';

const WINDOW = 20; // answers in the rolling "% correct"

function readUrl() {
  const params = new URLSearchParams(window.location.search);
  const num = (key) => (params.has(key) ? Number(params.get(key)) : null);
  return { student: num('student'), example: num('moment') };
}

// % correct over the last WINDOW answers, sampled to at most ~160 points so the line stays readable.
function yearSeries(attempts) {
  const step = Math.max(1, Math.ceil(attempts.length / 160));
  const rows = [];
  let right = 0;
  attempts.forEach((a, i) => {
    right += a.correct ? 1 : 0;
    if (i >= WINDOW) right -= attempts[i - WINDOW].correct ? 1 : 0;
    if (i >= Math.min(WINDOW, attempts.length) - 1 && (i % step === 0 || i === attempts.length - 1)) {
      rows.push({ x: i + 1, correct: right / Math.min(i + 1, WINDOW), date: a.date });
    }
  });
  return rows;
}

function Outcome({ rank, firstGroup }) {
  if (!rank) {
    return <span className="flex items-center gap-1.5 text-sm muted"><XCircle size={16} aria-hidden /> Not one of their next three topics</span>;
  }
  if (rank <= firstGroup) {
    return <span className="flex items-center gap-1.5 text-sm"><CheckCircle2 size={16} className="good" aria-hidden /> They started this next</span>;
  }
  return <span className="flex items-center gap-1.5 text-sm"><Clock size={16} className="good" aria-hidden /> They studied this soon after</span>;
}

function SuggestedVsActual({ example }) {
  const path = example.paths.pointer;
  return (
    <div className="flex flex-col gap-4">
      <ol className="m-0 p-0 list-none grid gap-4 lg:grid-cols-3">
        {path.concepts.map((c, step) => (
          <li key={c.id} className="card card-lg flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-full flex items-center justify-center font-semibold shrink-0" style={{ background: 'var(--accent)', color: '#fff' }}>
                {step + 1}
              </span>
              {c.unit && <span className="chip">{c.unit}</span>}
            </div>
            <h3 className="m-0 text-lg font-semibold leading-snug">{c.name}</h3>
            <p className="m-0 ink-2" style={{ lineHeight: 1.6 }}>{c.reason}</p>
            <div className="mt-auto pt-1">
              <Outcome rank={example.candidates[path.slots[step]].target_rank} firstGroup={example.first_group} />
            </div>
          </li>
        ))}
      </ol>
      <p className="m-0 ink-2">
        <span className="font-medium">What they actually did next:</span>{' '}
        {example.paths.actual.concepts.map((c) => c.name).join(' → ')}
      </p>
    </div>
  );
}

function YearChart({ detail, example }) {
  const data = useMemo(() => yearSeries(detail.attempts), [detail]);
  if (data.length < 2) return <p className="ink-2 m-0">This student has too few answers for a chart.</p>;
  const nearest = data.reduce((best, d) => (Math.abs(d.x - example.cut) < Math.abs(best.x - example.cut) ? d : best), data[0]);
  const dateOf = Object.fromEntries(data.map((d) => [d.x, d.date]));
  return (
    <div className="card card-lg">
      <LineChart title="Share of answers correct over the year" data={data} xLabel="Answers so far" yLabel={`% correct (last ${WINDOW} answers)`}
        series={[{ key: 'correct', name: '% correct', color: PLAN_COLOR.pointer }]} yDomain={[0, 1]}
        marker={nearest.x} markerLabel="This point" formatX={(x) => (dateOf[x] ? `Answer ${x} · ${fmtDate(dateOf[x])}` : `Answer ${x}`)} />
    </div>
  );
}

function StrengthsThenNow({ areas }) {
  const axes = areas.now.map((a) => ({
    key: `a${a.id}`, label: a.known ? a.name : `${a.name} (not practised)`,
    explain: a.known ? `Estimated from ${a.topics} topic${a.topics === 1 ? '' : 's'} they practised.` : 'They had not practised this area yet: a starting guess.',
  }));
  const values = (list) => Object.fromEntries(list.map((a) => [`a${a.id}`, a.value]));
  return (
    <div className="card card-lg">
      <SpiderChart title="Estimated understanding by area of maths" axes={axes} series={[
        { key: 'start', name: `At their first point (${fmtDate(areas.start_date)})`, color: 'var(--reference)', dashed: true, values: values(areas.start) },
        { key: 'now', name: 'At this point', color: PLAN_COLOR.pointer, values: values(areas.now) },
      ]} />
    </div>
  );
}

export default function Examples({ tutorMode }) {
  const [students, setStudents] = useState([]);
  const [studentId, setStudentId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [exampleId, setExampleId] = useState(null);
  const [example, setExample] = useState(null);
  const [graph, setGraph] = useState(null);
  const [errors, setErrors] = useState({});
  const initial = useRef(readUrl());
  const current = useRef(null);

  const selectExample = useCallback((id, sid) => {
    current.current = id;
    const stillCurrent = () => current.current === id;
    setExampleId(id);
    setExample(null);
    setGraph(null);
    setErrors({});
    const url = new URL(window.location.href);
    url.searchParams.set('student', sid);
    url.searchParams.set('moment', id);
    window.history.replaceState(null, '', url);
    const fail = (key) => (err) => stillCurrent() && setErrors((e) => ({ ...e, [key]: err.message }));
    api.example(id).then((x) => stillCurrent() && setExample(x)).catch(fail('example'));
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
    api.students()
      .then((s) => {
        setStudents(s);
        const wanted = s.find((x) => x.id === initial.current.student);
        const first = wanted ?? [...s].sort((a, b) => b.examples - a.examples)[0];
        if (first) selectStudent(first.id);
      })
      .catch((err) => setErrors({ load: err.message }));
  }, [selectStudent]);

  const summary = students.find((s) => s.id === studentId);

  return (
    <div className="page" style={{ maxWidth: 1280 }}>
      <header className="flex flex-col gap-3 pt-6">
        <div className="eyebrow">See it on real students</div>
        <h1 className="m-0 text-3xl font-semibold leading-tight" style={{ letterSpacing: '-0.02em' }}>Real students, real next steps</h1>
        <p className="section-lead">
          These are real students from Junyi Academy, an online maths platform, whom our recommender never saw while it was learning.
          Pick a student and a point in their year to see what it would have suggested, why, and what they really did next.
        </p>
      </header>
      {errors.load && <div className="card bad">Couldn't load the students ({errors.load}). Is the server running?</div>}

      <div className="grid gap-8 lg:grid-cols-[290px_minmax(0,1fr)] items-start">
        <aside>
          {students.length > 0
            ? <StudentList students={students} selected={studentId} onSelect={selectStudent} />
            : !errors.load && <div className="skeleton" style={{ height: 400 }} />}
        </aside>

        <main className="flex flex-col gap-12 min-w-0">
          {summary && <StudentHeader summary={summary} detail={detail} exampleId={exampleId} onSelectExample={(id) => selectExample(id, studentId)} />}
          {errors.example && <div className="card bad">Couldn't load this point: {errors.example}</div>}
          {!example && !errors.example && studentId !== null && <div className="skeleton" style={{ height: 320 }} />}
          {example && detail && (
            <>
              <Section id="year" title="Their year so far"
                lead={`How often they answered correctly, over their last ${WINDOW} answers at each point. The line marks the point you picked.`}>
                <YearChart detail={detail} example={example} />
              </Section>

              {example.areas && (
                <Section id="then-now" title="Strengths, then and now"
                  lead="Our estimate of how well they understood each area of maths at their first point and at the point you picked.">
                  <StrengthsThenNow areas={example.areas} />
                </Section>
              )}

              <Section id="suggested" eyebrow={`On ${fmtDate(example.date)}`} title="What we suggested next, and what they actually did"
                lead="Our recommender only looked at what came before this point. Each card says why it chose the topic and whether the student really went on to study it.">
                <SuggestedVsActual example={example} />
              </Section>

              <Section id="why" title="Why these three?"
                lead="Every suggestion balances five things. The further a point reaches out, the stronger that reason is for this topic.">
                <WhyTopic key={example.id} path={example.paths.pointer} />
              </Section>

              <Section id="compare" title="How it compares with simpler ways of choosing"
                lead="The same moment and the same options, chosen three ways. “Matched what they did” shows how many of the three this student really studied next.">
                <Compare paths={example.paths} keys={['pointer', 'markov', 'popularity']} axisKeys={EXAMPLE_AXES} />
              </Section>

              <Section id="connect" title="How the topics connect"
                lead="Each arrow points from a building block to a topic students almost always start after it. The suggested topics are outlined in blue.">
                <PrereqGraph example={example} graph={graph} error={errors.graph} />
              </Section>

              <Section id="tutor" title="Ask the tutor" lead="Hear this plan explained in plain words, or learn about one of its topics with a worked example.">
                <Tutor key={example.id} mode={tutorMode} whose="this plan" topics={example.paths.pointer.concepts}
                  ask={(body) => api.tutor(example.id, body)} />
              </Section>

              <TechnicalDetails>
                <CandidateTable example={example} />
              </TechnicalDetails>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
