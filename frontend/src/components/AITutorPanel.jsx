import React, { useState } from 'react';
import { Bot, Sparkles, BookOpen, Lightbulb, Loader2 } from 'lucide-react';

export default function AITutorPanel({ concept, recommendation, profile }) {
  const [aiExplanation, setAiExplanation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchAIReasoning = async (promptType) => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('http://localhost:5000/api/ai/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conceptId: concept.id,
          promptType
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setAiExplanation(data.explanation);
    } catch (err) {
      setError(err.message || 'Failed to connect to Gemini API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card rounded-2xl p-6 border border-indigo-500/30 glow-indigo">
      <div className="flex items-center justify-between pb-4 border-b border-slate-700/60 mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-500/20 text-indigo-400 rounded-xl border border-indigo-500/30">
            <Bot className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400">LLM Reasoning & Explanation</span>
            <h3 className="text-base font-bold text-white">Gemini AI Pedagogical Assistant</h3>
          </div>
        </div>
      </div>

      {/* AI Action Trigger Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-5">
        <button
          onClick={() => fetchAIReasoning('why_recommended')}
          disabled={loading}
          className="p-3 rounded-xl bg-gradient-to-r from-indigo-900/50 to-slate-800 border border-indigo-500/40 text-slate-200 text-xs font-semibold hover:border-indigo-400 transition flex items-center justify-center gap-2"
        >
          <Lightbulb className="w-4 h-4 text-amber-400" />
          Ask AI: "Why was this topic recommended?"
        </button>

        <button
          onClick={() => fetchAIReasoning('explain_concept')}
          disabled={loading}
          className="p-3 rounded-xl bg-gradient-to-r from-cyan-900/50 to-slate-800 border border-cyan-500/40 text-slate-200 text-xs font-semibold hover:border-cyan-400 transition flex items-center justify-center gap-2"
        >
          <BookOpen className="w-4 h-4 text-cyan-400" />
          Ask AI: "Tailored Concept Explanation"
        </button>
      </div>

      {/* Output Response View */}
      {loading && (
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800 text-center text-slate-400 text-xs flex items-center justify-center gap-2">
          <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
          Gemini AI is analyzing prerequisite state and user preferences...
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
          {error}
        </div>
      )}

      {aiExplanation && !loading && (
        <div className="p-4 rounded-xl bg-slate-900/90 border border-indigo-500/30 text-xs text-slate-200 leading-relaxed space-y-2">
          <div className="flex items-center gap-1.5 text-indigo-400 font-bold mb-2 text-[11px] uppercase tracking-wider">
            <Sparkles className="w-4 h-4" /> AI Pedagogical Insight
          </div>
          <div className="whitespace-pre-line text-slate-300">
            {aiExplanation}
          </div>
        </div>
      )}
    </div>
  );
}
