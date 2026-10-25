import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import Questionnaire from '../components/newstudent/Questionnaire';
import { CompareTable, KnownPanel, PathSteps, PipelineStrip, ShortlistTable } from '../components/newstudent/PlanView';

const EMPTY = { stage: 'junior', strands: {}, units: {}, current: [], practice: [] };

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

export default function NewStudent() {
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

  const change = (next, action) => {
    setLastAction(action);
    setAnswers(next);
  };

  const plan = hasSignal(answers) ? latest : null;

  if (error && !questionnaire) {
    return <div className="card bad">Couldn't load the questionnaire ({error}). This needs a bundle exported with <code>--content</code>.</div>;
  }
  if (!questionnaire) return <div className="skeleton" style={{ height: 400 }} />;

  return (
    <div className="grid gap-4 xl:grid-cols-[420px_minmax(0,1fr)] items-start">
      <div className="flex flex-col gap-4 xl:sticky xl:top-4">
        <Questionnaire questionnaire={questionnaire} answers={answers}
          onChange={(next) => change({ ...next, practice: next.stage === answers.stage ? answers.practice : [] }, 'Answers changed')}
          onExample={() => { const a = exampleAnswers(questionnaire); if (a) change(a, 'Loaded the example student'); }} />
        {plan && (
          <KnownPanel plan={plan} practiced={answers.practice.length}
            onResetPractice={() => change({ ...answers, practice: [] }, 'Cleared the practice')} />
        )}
      </div>
      <main className="flex flex-col gap-4 min-w-0">
        {error && <div className="card bad">{error}</div>}
        {!plan && !error && (
          <div className="card muted">
            Rate an area of maths as strong, mark a chapter, or name an exercise the student finished, and the path appears here.
          </div>
        )}
        {plan && (
          <>
            <PipelineStrip plan={plan} />
            <PathSteps plan={plan} prev={prev} lastAction={lastAction}
              onPractice={(c, correct) => change(
                { ...answers, practice: [...answers.practice, { concept: c.id, correct }] },
                `${correct ? 'Got' : 'Struggled with'} ${c.name}`,
              )} />
            <CompareTable plan={plan} />
            <ShortlistTable plan={plan} />
          </>
        )}
      </main>
    </div>
  );
}
