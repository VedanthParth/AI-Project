import { useState } from 'react';
import { PLAN_COLOR } from '../api';
import { axesOf, COMPARE_COLORS } from '../chartFormat';
import { inWords, METHODS, TOPIC_AXES } from '../copy';
import SpiderChart from './charts/SpiderChart';

/** Why each topic of a path is there: a spider chart of its five factors, one topic at a time. */
export function WhyTopic({ path }) {
  const [step, setStep] = useState(0);
  const topic = path.concepts[Math.min(step, path.concepts.length - 1)];
  const axes = axesOf(TOPIC_AXES);
  return (
    <div className="card card-lg grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-center">
      <div className="flex flex-col gap-3">
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Topic">
          {path.concepts.map((c, i) => (
            <button key={c.id} type="button" role="tab" aria-selected={i === step} className="btn" onClick={() => setStep(i)}
              style={i === step ? { background: 'var(--accent)', borderColor: 'var(--accent)', color: '#fff' } : undefined}>
              Topic {i + 1}
            </button>
          ))}
        </div>
        <SpiderChart title={`Why “${topic.name}”`} axes={axes}
          series={[{ key: 'topic', name: topic.name, color: PLAN_COLOR.pointer, values: topic.factors }]} />
      </div>
      <div className="flex flex-col gap-4">
        <h3 className="m-0 text-lg font-semibold leading-snug">{topic.name}</h3>
        <p className="m-0 ink-2" style={{ lineHeight: 1.6 }}>{topic.reason}</p>
        <dl className="m-0 flex flex-col gap-3">
          {axes.map((a) => (
            <div key={a.key}>
              <dt className="font-medium text-sm">{a.label}: <span className="ink-2 font-normal">{inWords(topic.factors[a.key])}</span></dt>
              <dd className="m-0 muted text-sm">{a.explain}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}

/** Ways of choosing, side by side: a spider chart of their profiles and a sentence on each. */
export function Compare({ paths, keys, axisKeys, title = 'How the plans compare', footer }) {
  const color = Object.fromEntries(keys.map((k, i) => [k, COMPARE_COLORS[i]]));
  return (
    <div className="card card-lg grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] items-center">
      <SpiderChart title={title} axes={axesOf(axisKeys)}
        series={keys.map((k) => ({ key: k, name: METHODS[k].name, color: color[k], values: paths[k].profile }))} />
      <div className="flex flex-col gap-4">
        <ul className="m-0 p-0 list-none flex flex-col gap-4">
          {keys.map((k) => (
            <li key={k} className="flex gap-3">
              <span className="inline-block w-3 h-3 rounded-full shrink-0 mt-1.5" style={{ background: color[k] }} aria-hidden />
              <div>
                <div className="font-medium">{METHODS[k].name}</div>
                <div className="ink-2 text-sm" style={{ lineHeight: 1.55 }}>{METHODS[k].what}</div>
                <div className="muted text-sm mt-1">{paths[k].concepts.map((c) => c.name).join(' → ')}</div>
              </div>
            </li>
          ))}
        </ul>
        {footer}
      </div>
    </div>
  );
}
