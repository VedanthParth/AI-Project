import React, { useState } from 'react';
import { HelpCircle, CheckCircle, XCircle, ArrowRight, RefreshCw, Award } from 'lucide-react';
import confetti from 'canvas-confetti';

export default function QuizModal({ concept, onSubmit, onClose }) {
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [results, setResults] = useState(null);

  if (!concept || !concept.quiz) return null;

  const handleSelectOption = (questionId, optionIndex) => {
    if (submitted) return;
    setSelectedAnswers(prev => ({
      ...prev,
      [questionId]: optionIndex
    }));
  };

  const handleSubmitQuiz = () => {
    const quizResults = concept.quiz.map(q => {
      const selected = selectedAnswers[q.id];
      const isCorrect = selected === q.correctIndex;
      return {
        questionId: q.id,
        selected,
        correctIndex: q.correctIndex,
        isCorrect
      };
    });

    const correctCount = quizResults.filter(r => r.isCorrect).length;
    setResults({ quizResults, correctCount, total: concept.quiz.length });
    setSubmitted(true);

    if (correctCount === concept.quiz.length) {
      confetti({ particleCount: 80, spread: 60, origin: { y: 0.6 } });
    }

    // Callback to backend BKT Engine update
    onSubmit(concept.id, quizResults);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6 rounded-2xl glass-card border border-cyan-500/40 text-slate-100 shadow-2xl relative">
        <div className="flex items-center justify-between pb-4 border-b border-slate-700/60 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/20 text-cyan-400 rounded-xl border border-cyan-500/30">
              <HelpCircle className="w-6 h-6" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-400">Diagnostic Practice Quiz</span>
              <h2 className="text-lg font-bold text-white">{concept.title}</h2>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-sm px-2 py-1">✕</button>
        </div>

        {/* Questions List */}
        <div className="space-y-6">
          {concept.quiz.map((q, idx) => {
            const isSelected = selectedAnswers[q.id] !== undefined;
            const res = results ? results.quizResults.find(r => r.questionId === q.id) : null;

            return (
              <div key={q.id} className="p-4 rounded-xl bg-slate-900/60 border border-slate-700/80">
                <div className="font-semibold text-sm text-slate-200 mb-3 flex items-start gap-2">
                  <span className="text-cyan-400 font-mono font-bold">Q{idx + 1}.</span>
                  <span>{q.question}</span>
                </div>

                <div className="space-y-2">
                  {q.options.map((opt, optIdx) => {
                    let optionStyle = 'bg-slate-800/60 border-slate-700 text-slate-300 hover:bg-slate-800';
                    const checked = selectedAnswers[q.id] === optIdx;

                    if (submitted) {
                      if (optIdx === q.correctIndex) {
                        optionStyle = 'bg-emerald-950/60 border-emerald-500 text-emerald-300 font-semibold';
                      } else if (checked && !res?.isCorrect) {
                        optionStyle = 'bg-rose-950/60 border-rose-500 text-rose-300';
                      }
                    } else if (checked) {
                      optionStyle = 'bg-cyan-950/60 border-cyan-400 text-cyan-300 glow-cyan';
                    }

                    return (
                      <button
                        key={optIdx}
                        disabled={submitted}
                        onClick={() => handleSelectOption(q.id, optIdx)}
                        className={`w-full text-left p-3 rounded-xl border text-xs transition-all flex items-center justify-between ${optionStyle}`}
                      >
                        <span>{opt}</span>
                        {submitted && optIdx === q.correctIndex && (
                          <CheckCircle className="w-4 h-4 text-emerald-400" />
                        )}
                        {submitted && checked && optIdx !== q.correctIndex && (
                          <XCircle className="w-4 h-4 text-rose-400" />
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* Explanation text after submission */}
                {submitted && (
                  <div className="mt-3 p-3 rounded-lg bg-slate-800/80 border border-slate-700 text-xs text-slate-300">
                    <strong className="text-cyan-400">Explanation:</strong> {q.explanation}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Footer actions */}
        <div className="mt-6 pt-4 border-t border-slate-700/60 flex items-center justify-between">
          {!submitted ? (
            <>
              <span className="text-xs text-slate-400">
                {Object.keys(selectedAnswers).length} of {concept.quiz.length} answered
              </span>
              <button
                disabled={Object.keys(selectedAnswers).length < concept.quiz.length}
                onClick={handleSubmitQuiz}
                className="px-5 py-2.5 text-xs font-bold text-slate-950 bg-gradient-to-r from-cyan-400 to-emerald-400 disabled:opacity-50 hover:from-cyan-300 hover:to-emerald-300 rounded-xl transition shadow-lg flex items-center gap-1.5"
              >
                Submit Answers & Update Mastery <ArrowRight className="w-4 h-4" />
              </button>
            </>
          ) : (
            <div className="w-full flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="w-5 h-5 text-emerald-400" />
                <span className="text-sm font-bold text-white">
                  Score: {results.correctCount} / {results.total} ({((results.correctCount / results.total) * 100).toFixed(0)}%)
                </span>
              </div>
              <button
                onClick={onClose}
                className="px-5 py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 rounded-xl border border-slate-700 transition"
              >
                Continue Learning Path
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
