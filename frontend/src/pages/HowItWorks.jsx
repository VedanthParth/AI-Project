import { ArrowDown, ArrowRight, BookOpenCheck, Brain, ListFilter, MessageSquare, Route, Sparkles } from 'lucide-react';
import { fmtNum } from '../api';
import Section from '../components/Section';
import TechnicalDetails from '../components/TechnicalDetails';

const FLOW = [
  { icon: MessageSquare, title: 'What we know about you', text: 'Your three answers, or a real practice history, and anything you practise here.', tech: 'Attempt history' },
  { icon: Brain, title: 'How well you know each topic', text: 'Every answer nudges our estimate up or down, topic by topic.', tech: 'Bayesian Knowledge Tracing' },
  { icon: ListFilter, title: 'A shortlist of what comes next', text: 'From all 1,326 exercises, the ones students usually take up after work like yours.', tech: 'Markov model of next topics' },
  { icon: Route, title: 'The best three, in order', text: 'The recommender looks at you and the whole shortlist together, then picks one topic at a time.', tech: 'Pointer network, fine-tuned with RL' },
  { icon: BookOpenCheck, title: 'Your plan, with reasons', text: 'Three topics, why each one is there, and how you might progress.', tech: 'Explanations and simulated practice' },
];

const DATA = [
  { value: 72758, label: 'students' },
  { value: 16217311, label: 'answers to maths questions' },
  { value: 1326, label: 'different exercises' },
  { value: 12, label: 'months of real learning, 2018–19' },
];

const DIFFERENT = [
  ['It learns the order from real students', 'Instead of a fixed syllabus, it learned from what tens of thousands of students actually did next, and in what order.'],
  ['It respects building blocks', 'It works out which topics students almost always do first, and tries to put those before the topics that need them.'],
  ['It is tuned for learning', 'After learning from real students, it practised in a simulated classroom, where it was rewarded for topics expected to help more and for not skipping building blocks.'],
  ['It explains itself', 'Every suggestion comes with a reason in plain words and a chart of what tipped the balance.'],
  ['It works for someone new', 'Three quick questions are enough to make a first plan, and it adapts as soon as you start practising.'],
];

const LIMITS = [
  'It learned from Junyi Academy, a Taiwanese platform. Its chapters are matched to Indian classes only roughly, and exercise names are translated by machine.',
  'It recommends topics students like you went on to study. That is a strong guide, but not proof that a topic is the best one for learning.',
  'It only suggests new topics. Topics worth reviewing are picked by a simple rule: the ones you have worked on that still look shaky.',
  'With only three answers to go on, the plan is a first guess. It gets better as you tell it how each topic goes.',
];

export default function HowItWorks() {
  return (
    <div className="page">
      <header className="flex flex-col gap-3 pt-6">
        <div className="eyebrow">How it works</div>
        <h1 className="m-0 text-3xl font-semibold leading-tight" style={{ letterSpacing: '-0.02em', maxWidth: '24ch' }}>
          A study planner that learned from real students
        </h1>
        <p className="section-lead">
          What to study next, and in what order, makes a big difference in maths: some topics are building blocks for others. This app
          suggests the next three topics for a student and explains why, using what it learned from tens of thousands of real students.
        </p>
      </header>

      <Section id="flow" title="From your answers to your plan" lead="Five simple steps. This is everything the app does, start to finish.">
        <ol className="m-0 p-0 list-none flex flex-col lg:flex-row gap-3 lg:items-stretch">
          {FLOW.map((step, i) => {
            const Icon = step.icon;
            return (
              <li key={step.title} className="contents">
                <div className="card card-lg flex-1 flex flex-col gap-3">
                  <span className="w-10 h-10 rounded-full flex items-center justify-center" style={{ background: 'var(--accent-soft)', color: 'var(--accent)' }}>
                    <Icon size={20} aria-hidden />
                  </span>
                  <div className="font-semibold leading-snug">{step.title}</div>
                  <p className="m-0 ink-2 text-sm" style={{ lineHeight: 1.6 }}>{step.text}</p>
                  <div className="muted text-xs mt-auto">{step.tech}</div>
                </div>
                {i < FLOW.length - 1 && (
                  <div className="flex items-center justify-center muted" aria-hidden>
                    <ArrowRight size={20} className="hidden lg:block" />
                    <ArrowDown size={20} className="lg:hidden" />
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      </Section>

      <Section id="data" title="What it learned from" lead="A year of real practice on Junyi Academy, an online maths platform used by school students.">
        <dl className="m-0 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {DATA.map((d) => (
            <div key={d.label} className="card card-lg flex flex-col-reverse gap-1">
              <dt className="muted text-sm">{d.label}</dt>
              <dd className="m-0 text-3xl font-semibold">{fmtNum(d.value)}</dd>
            </div>
          ))}
        </dl>
        <p className="m-0 muted text-sm">
          Students were split into groups: the recommender learned from one group, and every student on the “See it on real students”
          page comes from a group it never saw.
        </p>
      </Section>

      <Section id="different" title="What makes it more than a list of topics">
        <ul className="m-0 p-0 list-none grid gap-4 md:grid-cols-2">
          {DIFFERENT.map(([title, text]) => (
            <li key={title} className="card card-lg flex gap-3">
              <Sparkles size={18} className="shrink-0 mt-0.5" style={{ color: 'var(--accent)' }} aria-hidden />
              <div>
                <div className="font-semibold mb-1">{title}</div>
                <p className="m-0 ink-2 text-sm" style={{ lineHeight: 1.6 }}>{text}</p>
              </div>
            </li>
          ))}
        </ul>
      </Section>

      <Section id="limits" title="What it can't do (yet)" lead="Being honest about the limits matters as much as the features.">
        <ul className="m-0 card card-lg flex flex-col gap-3 pl-10">
          {LIMITS.map((text) => <li key={text} className="ink-2" style={{ lineHeight: 1.6 }}>{text}</li>)}
        </ul>
      </Section>

      <TechnicalDetails />
    </div>
  );
}
