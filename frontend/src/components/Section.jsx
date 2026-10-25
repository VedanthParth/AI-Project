// One idea per section: a heading, a sentence saying what it shows, then the content.
export default function Section({ id, eyebrow, title, lead, children }) {
  return (
    <section className="section" aria-labelledby={`${id}-title`}>
      <div className="flex flex-col gap-2">
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2 className="section-title" id={`${id}-title`}>{title}</h2>
        {lead && <p className="section-lead">{lead}</p>}
      </div>
      {children}
    </section>
  );
}
