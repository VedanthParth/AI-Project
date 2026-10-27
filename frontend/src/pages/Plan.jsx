import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowRight, Check, Gauge, ListFilter, MessageSquare, RotateCcw, Route, X } from 'lucide-react';
import { api, fmtNum, PLAN_COLOR } from '../api';
import { HOW_IT_WORKS, METHODS, PLAN_AXES } from '../copy';
import Questionnaire from '../components/newstudent/Questionnaire';
import Section from '../components/Section';
import SpiderChart from '../components/charts/SpiderChart';
import { Compare, WhyTopic } from '../components/Explain';
import { COMPARE_COLORS } from '../chartFormat';
import LineChart from '../components/charts/LineChart';
import Tutor from '../components/Tutor';
import TechnicalDetails from '../components/TechnicalDetails';

const EMPTY = { stage: 'junior', strands: {}, units: {}, current: [], practice: [] };
const STEP_ICONS = [MessageSquare, Gauge, ListFilter, Route];

// A Class 7–9 student who is fine with linear equations, struggles with quadratics and has just started them.
function exampleAnswers(q) {
  const unit = (name) => q.units.find((u) => u.stage === 'junior' && u.name === name);
  const algebra = q.strands.find((s) => s.stage === 'junior' && s.name === 'Algebra');
  const linear = unit('Linear Equations in One Variable');
  const quadratic = unit('Quadratic Equations');
  if (!algebra || !linear || !quadratic) return null;
  return {
    stage: 'junior',
    strands: { [algebra.id]: 'okay' },
    units: { [linear.id]: 'good', [quadratic.id]: 'bad' },
    current: [quadratic.exercises[Math.min(3, quadratic.exercises.length - 1)].id],
    practice: [],
  };
}

function hasSignal(a) {
  return a.current.length > 0 || Object.keys(a.units).length > 0 || Object.values(a.strands).includes('strong') || a.practice.length > 0;
}

function HowItWorksStrip() {
  return (
    <ol className="m-0 p-0 list-none grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {HOW_IT_WORKS.map((step, i) => {
        const Icon = STEP_ICONS[i];
        return (
          <li key={step.title} className="card card-lg flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <span className="w-9 h-9 rounded-full flex items-center justify-center shrink-0" style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}>
                <Icon size={18} aria-hidden />
              </span>
              <span className="muted text-sm">Step {i + 1}</span>
            </div>
            <div className="font-semibold">{step.title}</div>
            <p className="m-0 ink-2 text-sm" style={{ lineHeight: 1.6 }}>{step.text}</p>
          </li>
        );
      })}
    </ol>
  );
}

const STYLE = 'pointer'; // the recommender: the pointer network after its RL fine-tuning stage

function changes(plan, prev, style) {
  if (!prev?.paths[style]) return null;
  const now = plan.paths[style].concepts.map((c) => c.id);
  const before = prev.paths[style].concepts.map((c) => c.id);
  if (now.join() === before.join()) return { same: true };
  const names = Object.fromEntries([...plan.paths[style].concepts, ...prev.paths[style].concepts].map((c) => [c.id, c.name]));
  return { added: now.filter((c) => !before.includes(c)).map((c) => names[c]), dropped: before.filter((c) => !now.includes(c)).map((c) => names[c]) };
}

function TopicCards({ plan, prev, style, lastAction, onPractice }) {
  const diff = changes(plan, prev, style);
  return (
    <div className="flex flex-col gap-4">
      {diff && lastAction && (
        <div className="rounded-xl px-4 py-3" style={{ background: 'var(--accent-soft)' }} role="status">
          <span className="font-medium">{lastAction}.</span>{' '}
          {diff.same ? 'Your next three topics stay the same.' : (
            <>
              {diff.added.length > 0 && <>New in your plan: <b>{diff.added.join(', ')}</b>. </>}
              {diff.dropped.length > 0 && <>No longer in it: {diff.dropped.join(', ')}.</>}
            </>
          )}
        </div>
      )}
      <ol className="m-0 p-0 list-none grid gap-4 lg:grid-cols-3">
        {plan.paths[style].concepts.map((c, step) => (
          <li key={c.id} className="card card-lg flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <span className="w-8 h-8 rounded-full flex items-center justify-center font-semibold shrink-0" style={{ background: 'var(--accent)', color: '#fff' }}>
                {step + 1}
              </span>
              <span className="chip">{c.unit}</span>
            </div>
            <h3 className="m-0 text-lg font-semibold leading-snug">{c.name}</h3>
            <p className="m-0 ink-2" style={{ lineHeight: 1.6 }}>
              {c.unmet_prereqs.length > 0 && <AlertTriangle size={15} className="inline mr-1 align-[-2px]" style={{ color: 'var(--series-4)' }} aria-hidden />}
              {c.reason}
            </p>
            <div className="flex gap-2 mt-auto">
              <button type="button" className="btn flex-1 justify-center" onClick={() => onPractice(c, true)}>
                <Check size={16} className="good" aria-hidden /> Got it
              </button>
              <button type="button" className="btn flex-1 justify-center" onClick={() => onPractice(c, false)}>
                <X size={16} className="bad" aria-hidden /> Found it hard
              </button>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Progress({ plan, style }) {
  const keys = [style, 'markov', 'popularity'];
  const colors = Object.fromEntries(keys.map((k, i) => [k, COMPARE_COLORS[i]]));
  const curves = plan.simulation.paths;
  const data = curves[style].curve.map((_, i) => ({ x: i, ...Object.fromEntries(keys.map((k) => [k, curves[k].curve[i]])) }));
  const per = plan.simulation.attempts_per_concept;
  return (
    <div className="card card-lg">
      <LineChart title="Predicted scores while practising each plan" data={data} xLabel="Questions practised" yLabel="Predicted score"
        series={keys.map((k) => ({ key: k, name: METHODS[k].name, color: colors[k] }))}
        formatX={(x) => (x === 0 ? 'Before practising' : `${x} question${x > 1 ? 's' : ''} (topic ${Math.ceil(x / per)})`)} />
    </div>
  );
}

function Strengths({ plan }) {
  const practised = plan.practice_trace.length > 0;
  const axes = plan.areas.now.map((a) => ({
    key: `a${a.id}`, label: a.known ? a.name : `${a.name} (no answers yet)`,
    explain: a.known ? `Estimated from ${a.topics} topic${a.topics === 1 ? '' : 's'} you told us about or practised.` : 'A starting guess: you have not told us about this area yet.',
  }));
  const values = (list) => Object.fromEntries(list.map((a) => [`a${a.id}`, a.value]));
  const series = practised
    ? [
        { key: 'before', name: 'Before practising here', color: 'var(--reference)', dashed: true, values: values(plan.areas.before) },
        { key: 'now', name: 'Now', color: PLAN_COLOR.pointer, values: values(plan.areas.now) },
      ]
    : [{ key: 'now', name: 'From your answers', color: PLAN_COLOR.pointer, values: values(plan.areas.now) }];
  const trace = plan.practice_trace.slice(-3);
  const traceColors = [PLAN_COLOR.pointer, PLAN_COLOR.markov, PLAN_COLOR.rl];
  const longest = Math.max(0, ...trace.map((t) => t.points.length));
  const traceData = Array.from({ length: longest }, (_, i) => ({ x: i, ...Object.fromEntries(trace.map((t) => [`t${t.id}`, t.points[i] ?? null])) }));
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="card card-lg flex flex-col gap-3">
        <h3 className="m-0 font-semibold">By area of maths</h3>
        <SpiderChart title="Estimated understanding by area of maths" axes={axes} series={series} height={320} />
      </div>
      <div className="card card-lg flex flex-col gap-4">
        <h3 className="m-0 font-semibold">While you practise</h3>
        {practised ? (
          <LineChart title="Estimated understanding of each practised topic" data={traceData} xLabel="Attempts" yLabel="Estimated understanding"
            yDomain={[0, 1]} series={trace.map((t, i) => ({ key: `t${t.id}`, name: t.name, color: traceColors[i] }))}
            formatX={(x) => (x === 0 ? 'Before practising' : `After attempt ${x}`)} height={260} />
        ) : (
          <p className="m-0 ink-2" style={{ lineHeight: 1.6 }}>
            Press “Got it” or “Found it hard” on a topic above, and this chart shows how our estimate of your understanding changes
            with each attempt. The plan updates too.
          </p>
        )}
        {plan.review.length > 0 && (
          <div className="rounded-xl p-4" style={{ background: 'var(--surface-2)' }}>
            <div className="font-medium mb-1">Worth reviewing first</div>
            <p className="m-0 muted text-sm mb-2">Topics you have worked on that still look shaky. The recommender only suggests new topics, so these are listed separately.</p>
            <ul className="m-0 pl-5 text-sm">{plan.review.map((r) => <li key={r.id}>{r.name}</li>)}</ul>
          </div>
        )}
      </div>
    </div>
  );
}

function Technical({ plan, style }) {
  const p = plan.pipeline;
  const path = plan.paths[style];
  return (
    <TechnicalDetails>
      <div>
        <div className="font-medium mb-2">This plan, step by step</div>
        <p className="m-0 ink-2">
          {fmtNum(p.catalogue)} exercises → {fmtNum(p.stage_concepts)} at this class group → {fmtNum(p.eligible)} not done yet →
          a shortlist of {p.shortlist_size} from the Markov model → 3 ordered by {p.models[style] ?? p.models.pointer}. Your answers
          became {p.history_attempts} attempts on {p.history_concepts} topics.
        </p>
      </div>
      {path.steps && (
        <div>
          <div className="font-medium mb-2">The recommender's probability for its top five options at each step</div>
          <div className="grid gap-4 md:grid-cols-3">
            {path.steps.map((row, step) => (
              <ul key={step} className="m-0 p-0 list-none flex flex-col gap-1 text-xs">
                <li className="font-medium">Step {step + 1}</li>
                {row.map((prob, slot) => ({ prob, slot })).sort((a, b) => b.prob - a.prob).slice(0, 5).map(({ prob, slot }) => (
                  <li key={slot} className="grid grid-cols-[minmax(0,1fr)_40px] gap-2">
                    <span className={slot === path.concepts[step].slot ? 'font-medium truncate' : 'muted truncate'}>{plan.shortlist[slot].name}</span>
                    <span className="num text-right">{Math.round(prob * 100)}%</span>
                  </li>
                ))}
              </ul>
            ))}
          </div>
        </div>
      )}
    </TechnicalDetails>
  );
}

export default function Plan({ tutorMode }) {
  const [questionnaire, setQuestionnaire] = useState(null);
  const [answers, setAnswers] = useState(EMPTY);
  const [latest, setPlan] = useState(null);
  const [prev, setPrev] = useState(null);
  const [lastAction, setLastAction] = useState(null);
  const [error, setError] = useState(null);
  const request = useRef(0);
  const planRef = useRef(null);

  useEffect(() => {
    api.questionnaire().then(setQuestionnaire).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!hasSignal(answers)) return undefined;
    const id = ++request.current;
    const timer = setTimeout(() => {
      api.plan(answers)
        .then((p) => {
          if (id !== request.current) return;
          setPrev(planRef.current);
          planRef.current = p;
          setPlan(p);
          setError(null);
        })
        .catch((err) => id === request.current && setError(err.message));
    }, 200);
    return () => clearTimeout(timer);
  }, [answers]);

  const plan = hasSignal(answers) ? latest : null;
  const change = (next, action) => {
    setLastAction(action);
    setAnswers(next);
  };

  return (
    <div className="page">
      <header className="flex flex-col gap-6 pt-6">
        <div className="flex flex-col gap-3">
          <div className="eyebrow">Plan my next topics</div>
          <h1 className="m-0 text-3xl font-semibold leading-tight" style={{ letterSpacing: '-0.02em', maxWidth: '22ch' }}>
            Not sure what to study next in maths?
          </h1>
          <p className="section-lead">
            Answer three quick questions. We suggest the next three topics, in the order that makes sense for you, and show the reasons
            behind every suggestion.
          </p>
        </div>
        <HowItWorksStrip />
      </header>

      <Section id="questions" title="Tell us about yourself" lead="There are no wrong answers. The more you tell us, the better the plan fits.">
        {error && !questionnaire && <div className="card bad">Couldn't load the questions ({error}).</div>}
        {questionnaire ? (
          <Questionnaire questionnaire={questionnaire} answers={answers}
            onChange={(next) => change({ ...next, practice: next.stage === answers.stage ? answers.practice : [] }, 'You changed your answers')}
            onExample={() => { const a = exampleAnswers(questionnaire); if (a) change(a, 'Loaded the example student'); }} />
        ) : !error && <div className="skeleton" style={{ height: 320 }} />}
      </Section>

      {error && questionnaire && <div className="card bad">{error}</div>}
      {!plan && questionnaire && (
        <div className="card card-lg flex items-center gap-3 ink-2">
          <ArrowRight size={18} aria-hidden /> Answer at least one question above, or try the example student, and your plan appears here.
        </div>
      )}

      {plan && (
        <>
          <Section id="plan" eyebrow="Your plan" title="Your next three topics"
            lead="Do them in this order. Each card says why it is there. Tell us how each one goes and the plan adjusts.">
            <p className="m-0 muted text-sm" style={{ maxWidth: '68ch' }}>{METHODS.pointer.what}</p>
            <TopicCards plan={plan} prev={prev} style={STYLE} lastAction={lastAction}
              onPractice={(c, correct) => change(
                { ...answers, practice: [...answers.practice, { concept: c.id, correct }] },
                correct ? `You got “${c.name}”` : `You found “${c.name}” hard`,
              )} />
          </Section>

          <Section id="why" title="Why these three?"
            lead="Every suggestion balances five things. The further a point reaches out, the stronger that reason is for this topic.">
            <WhyTopic key={plan.paths[STYLE].concepts.map((c) => c.id).join()} path={plan.paths[STYLE]} />
          </Section>

          <Section id="compare" title="How your plan compares with simpler ways of choosing"
            lead="Two simple ways to pick topics, next to ours, for the same answers. A good plan reaches out on every side, not just one.">
            <Compare paths={plan.paths} keys={[STYLE, 'markov', 'popularity']} axisKeys={PLAN_AXES} />
          </Section>

          <Section id="progress" title="How you might progress"
            lead="A separate program that imitates a student predicts how your scores on nearby topics would change while you practise each plan, three questions per topic.">
            <Progress plan={plan} style={STYLE} />
          </Section>

          <Section id="strengths" title="Your strengths"
            lead="Our estimate of how well you understand each area, from your answers and anything you practise here.">
            <Strengths plan={plan} />
          </Section>

          <Section id="tutor" title="Ask the tutor" lead="Get your plan explained in plain words, or learn about one of its topics with a worked example.">
            <Tutor mode={tutorMode} topics={plan.paths[STYLE].concepts}
              ask={(body) => api.explain({ ...answers, ...body })} />
          </Section>

          <Technical plan={plan} style={STYLE} />
          {answers.practice.length > 0 && (
            <button type="button" className="btn self-start" onClick={() => change({ ...answers, practice: [] }, 'You cleared your practice')}>
              <RotateCcw size={14} aria-hidden /> Clear my practice
            </button>
          )}
        </>
      )}
    </div>
  );
}
