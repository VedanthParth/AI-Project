import React from 'react';
import { Lock, CheckCircle2, Brain, ArrowDownRight, ArrowRight } from 'lucide-react';

export default function ConceptGraph({ concepts, masteries, completedConcepts, recommendedId, onSelectConcept }) {
  return (
    <div className="glass-card rounded-2xl p-6 border border-slate-700/60">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            Prerequisite Knowledge Graph (DAG)
          </h3>
          <p className="text-xs text-slate-400">Strict dependency order ensures concepts build logically upon unlocked foundations.</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Mastered (&gt;75%)
          </div>
          <div className="flex items-center gap-1.5 text-cyan-400">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 glow-cyan"></span> Recommended
          </div>
          <div className="flex items-center gap-1.5 text-slate-500">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-600"></span> Locked Prereq
          </div>
        </div>
      </div>

      {/* DAG Flowchart Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 relative">
        {concepts.map((concept, index) => {
          const mastery = masteries[concept.id] || 0.25;
          const isCompleted = completedConcepts.includes(concept.id) || mastery >= 0.75;
          const isRecommended = concept.id === recommendedId;

          // Check if prerequisites are unlocked
          const prereqsMet = concept.prerequisites.every(p => 
            (masteries[p] || 0) >= 0.70 || completedConcepts.includes(p)
          );

          const isLocked = !prereqsMet && !isCompleted;

          return (
            <div
              key={concept.id}
              onClick={() => !isLocked && onSelectConcept(concept)}
              className={`p-4 rounded-xl border transition-all relative cursor-pointer flex flex-col justify-between ${
                isRecommended
                  ? 'bg-gradient-to-b from-cyan-950/60 to-slate-900 border-cyan-400 glow-cyan ring-1 ring-cyan-400'
                  : isCompleted
                  ? 'bg-emerald-950/20 border-emerald-500/40 text-slate-300 hover:border-emerald-400'
                  : isLocked
                  ? 'bg-slate-900/40 border-slate-800 text-slate-600 cursor-not-allowed opacity-75'
                  : 'bg-slate-800/40 border-slate-700/80 text-slate-300 hover:border-slate-500'
              }`}
            >
              {/* Card Header */}
              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Concept 0{index + 1}
                  </span>
                  {isCompleted ? (
                    <span className="px-2 py-0.5 text-[10px] font-bold bg-emerald-500/20 text-emerald-400 rounded-full border border-emerald-500/30 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Mastered
                    </span>
                  ) : isRecommended ? (
                    <span className="px-2 py-0.5 text-[10px] font-bold bg-cyan-500/20 text-cyan-300 rounded-full border border-cyan-400/40 animate-pulse">
                      NEXT TARGET
                    </span>
                  ) : isLocked ? (
                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-slate-800 text-slate-500 rounded-full flex items-center gap-1">
                      <Lock className="w-3 h-3" /> Locked
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-slate-700 text-slate-300 rounded-full">
                      Ready
                    </span>
                  )}
                </div>

                <h4 className={`font-bold text-sm mb-1 ${isRecommended ? 'text-cyan-300' : isCompleted ? 'text-emerald-300' : 'text-slate-200'}`}>
                  {concept.title}
                </h4>
                <p className="text-xs text-slate-400 line-clamp-2 mb-3">
                  {concept.description}
                </p>
              </div>

              {/* Prerequisites Indicator */}
              {concept.prerequisites.length > 0 && (
                <div className="mb-3 text-[11px] text-slate-500 flex items-center gap-1">
                  <ArrowRight className="w-3 h-3 text-slate-600" />
                  Prereq: {concept.prerequisites.join(', ')}
                </div>
              )}

              {/* Knowledge Tracing Progress Bar */}
              <div>
                <div className="flex justify-between items-center text-[11px] mb-1">
                  <span className="text-slate-400">BKT Knowledge Mastery P(L):</span>
                  <span className={`font-mono font-bold ${mastery >= 0.75 ? 'text-emerald-400' : 'text-cyan-400'}`}>
                    {(mastery * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700/50">
                  <div
                    className={`h-full transition-all duration-500 rounded-full ${
                      mastery >= 0.75
                        ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                        : 'bg-gradient-to-r from-cyan-500 to-indigo-500'
                    }`}
                    style={{ width: `${Math.min(100, Math.max(5, mastery * 100))}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
