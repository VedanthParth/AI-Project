import { useState } from 'react';
import { BookOpen, MessageCircle, Sparkles } from 'lucide-react';

/**
 * The tutor: explains a plan or teaches one of its topics.
 * ask(body) returns { source, text }; mode is "gemini" or "template" (from /api/info).
 */
export default function Tutor({ ask, topics, mode, whose = 'your plan' }) {
  const [topic, setTopic] = useState(String(topics[0]?.id ?? ''));
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const chosen = topics.some((t) => String(t.id) === topic) ? topic : String(topics[0]?.id ?? '');

  const run = async (body) => {
    setBusy(true);
    setAnswer(null);
    try {
      setAnswer(await ask(body));
    } catch (err) {
      setAnswer({ error: err.message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card card-lg flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="chip">
          <Sparkles size={12} aria-hidden />
          {mode === 'gemini' ? 'Connected to Gemini' : 'Quick answers (Gemini not connected)'}
        </span>
        {mode !== 'gemini' && (
          <span className="muted text-xs">Add GEMINI_API_KEY to backend/.env and restart the server for fuller answers.</span>
        )}
      </div>
      <div className="flex flex-wrap gap-3">
        <button type="button" className="btn btn-primary" disabled={busy} onClick={() => run({ kind: 'why' })}>
          <MessageCircle size={16} aria-hidden /> Explain {whose}
        </button>
        <div className="flex gap-2 min-w-0 flex-1">
          <select className="input text-sm min-w-0" value={chosen} onChange={(e) => setTopic(e.target.value)} aria-label="Topic to learn about">
            {topics.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
          <button type="button" className="btn shrink-0" disabled={busy || !chosen}
            onClick={() => run({ kind: 'concept', concept_id: Number(chosen) })}>
            <BookOpen size={16} aria-hidden /> Teach me this topic
          </button>
        </div>
      </div>
      {busy && <div className="skeleton" style={{ height: 96 }} />}
      {answer?.error && <p className="bad m-0">The tutor couldn't answer: {answer.error}</p>}
      {answer?.text && (
        <div className="rounded-xl p-4" style={{ background: 'var(--surface-2)' }}>
          <div className="eyebrow mb-2">{answer.source === 'gemini' ? 'Answer from Gemini' : 'Quick answer'}</div>
          <p className="m-0" style={{ whiteSpace: 'pre-line', lineHeight: 1.65 }}>{answer.text}</p>
        </div>
      )}
    </div>
  );
}
