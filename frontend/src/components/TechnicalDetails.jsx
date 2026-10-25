import { TECHNICAL } from '../copy';

// The only place technical names appear: a collapsed panel for readers who want to know how it is built.
export default function TechnicalDetails({ children }) {
  return (
    <details className="card card-lg">
      <summary className="cursor-pointer font-medium">For the technically curious</summary>
      <div className="flex flex-col gap-6 mt-4 text-sm">
        <dl className="m-0 grid gap-3 sm:grid-cols-[220px_minmax(0,1fr)]">
          {TECHNICAL.map(([term, text]) => (
            <div key={term} className="contents">
              <dt className="font-medium">{term}</dt>
              <dd className="m-0 ink-2">{text}</dd>
            </div>
          ))}
        </dl>
        {children}
      </div>
    </details>
  );
}
