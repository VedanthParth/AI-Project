import React from 'react';
import { Sparkles, Check, Clock, X, HelpCircle, ShieldAlert } from 'lucide-react';

export default function ConsentTrialBanner({ proposal, onRespond }) {
  if (!proposal) return null;

  return (
    <div className="p-5 rounded-2xl bg-gradient-to-r from-amber-950/40 via-indigo-950/40 to-slate-900 border border-amber-500/40 glow-indigo shadow-2xl relative overflow-hidden">
      <div className="absolute -right-10 -bottom-10 w-40 h-40 bg-amber-500/10 rounded-full blur-2xl pointer-events-none"></div>

      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 relative z-10">
        <div className="flex items-start gap-3">
          <div className="p-3 bg-amber-500/20 text-amber-400 rounded-xl border border-amber-500/30 shrink-0">
            <Sparkles className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="px-2 py-0.5 text-[10px] font-bold bg-amber-500/20 text-amber-300 rounded-full border border-amber-400/30 uppercase tracking-wider">
                Consent-Based Exploration Proposal
              </span>
              <span className="text-xs text-slate-400 flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-amber-400" /> 3-Session Trial
              </span>
            </div>
            <h4 className="text-base font-bold text-white mb-1">
              Proposed Learning Trial: Try <span className="text-amber-300 underline uppercase">{proposal.targetFormat}</span> Format
            </h4>
            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              {proposal.reasoning}
            </p>
          </div>
        </div>

        {/* Interactive Student Consent Actions */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            onClick={() => onRespond('accept', proposal.targetFormat, proposal.durationSessions)}
            className="px-4 py-2 text-xs font-bold text-slate-950 bg-amber-400 hover:bg-amber-300 rounded-xl transition-all shadow-md flex items-center gap-1.5"
          >
            <Check className="w-4 h-4" /> Accept Trial (3 Sessions)
          </button>
          
          <button
            onClick={() => onRespond('postpone')}
            className="px-3 py-2 text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-xl border border-slate-700 transition"
          >
            Postpone
          </button>

          <button
            onClick={() => onRespond('decline')}
            className="px-3 py-2 text-xs font-semibold text-rose-400 hover:bg-rose-950/40 rounded-xl border border-rose-800/50 transition flex items-center gap-1"
          >
            <X className="w-3.5 h-3.5" /> Keep My Habit
          </button>
        </div>
      </div>
    </div>
  );
}
