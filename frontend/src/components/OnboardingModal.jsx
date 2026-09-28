import React from 'react';
import { User, Video, BookOpen, Dumbbell, Clock, Compass, ShieldCheck } from 'lucide-react';

export default function OnboardingModal({ profile, onSave, onClose }) {
  const [preferredFormat, setPreferredFormat] = React.useState(profile.preferredFormat || 'video');
  const [sessionLength, setSessionLength] = React.useState(profile.sessionLength || 15);
  const [learningStyle, setLearningStyle] = React.useState(profile.learningStyle || 'explanation');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave({ preferredFormat, sessionLength: Number(sessionLength), learningStyle });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
      <div className="w-full max-w-xl p-6 rounded-2xl glass-card border border-cyan-500/30 text-slate-100 shadow-2xl relative">
        <div className="flex items-center justify-between pb-4 border-b border-slate-700/60 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/20 text-cyan-400 rounded-xl border border-cyan-500/30">
              <User className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold gradient-text">Stable Study Preferences</h2>
              <p className="text-xs text-slate-400">Set your baseline habits. We will NEVER change these without your consent.</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-sm px-2 py-1 rounded">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* 1. Preferred Resource Format */}
          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
              <Compass className="w-4 h-4 text-cyan-400" /> Preferred Resource Format
            </label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { id: 'video', label: 'Video Lessons', icon: Video, desc: 'Visual step-by-step videos' },
                { id: 'text', label: 'Written Text', icon: BookOpen, desc: 'Structured documentation' },
                { id: 'practice', label: 'Hands-on Practice', icon: Dumbbell, desc: 'Interactive problems first' }
              ].map(item => {
                const IconComponent = item.icon;
                const active = preferredFormat === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setPreferredFormat(item.id)}
                    className={`p-3.5 rounded-xl text-left border transition-all flex flex-col items-start gap-2 ${
                      active 
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 glow-cyan' 
                        : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <IconComponent className={`w-5 h-5 ${active ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <div>
                      <div className="font-semibold text-sm">{item.label}</div>
                      <div className="text-[11px] text-slate-400 mt-0.5 leading-tight">{item.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Session Length */}
          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4 text-indigo-400" /> Micro-Learning Session Duration
            </label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { minutes: 10, label: '10 Mins / Session', tag: 'Bite-sized' },
                { minutes: 15, label: '15 Mins / Session', tag: 'Standard' },
                { minutes: 30, label: '30 Mins / Session', tag: 'Deep Focus' }
              ].map(item => {
                const active = Number(sessionLength) === item.minutes;
                return (
                  <button
                    key={item.minutes}
                    type="button"
                    onClick={() => setSessionLength(item.minutes)}
                    className={`p-3 rounded-xl text-center border transition-all ${
                      active 
                        ? 'bg-indigo-500/20 border-indigo-400 text-indigo-300 glow-indigo' 
                        : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <div className="font-bold text-sm">{item.label}</div>
                    <div className="text-[11px] text-slate-400 mt-1">{item.tag}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 3. Learning Pathway Style */}
          <div>
            <label className="block text-sm font-semibold text-slate-300 mb-2 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400" /> Pedagogical Approach Preference
            </label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { id: 'explanation', title: 'Explanation-First', desc: 'Read/watch theory concepts before quiz exercises.' },
                { id: 'practice_first', title: 'Practice-First', desc: 'Attempt diagnostic quiz upfront, review theory if stuck.' }
              ].map(item => {
                const active = learningStyle === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setLearningStyle(item.id)}
                    className={`p-3.5 rounded-xl text-left border transition-all ${
                      active 
                        ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300 glow-emerald' 
                        : 'bg-slate-800/40 border-slate-700 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    <div className="font-semibold text-sm text-slate-200">{item.title}</div>
                    <div className="text-xs text-slate-400 mt-1">{item.desc}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Consent Guarantee Note */}
          <div className="p-3 bg-slate-900/80 border border-slate-700/80 rounded-xl text-xs text-slate-400 flex items-start gap-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <span>
              <strong>Student Control Policy:</strong> Your preferences remain fixed. If Knowledge Tracing detects that a different style could accelerate learning, the system will propose a 3-session trial for your approval.
            </span>
          </div>

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-slate-400 hover:text-white transition"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-5 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-cyan-500 to-indigo-600 rounded-xl hover:from-cyan-400 hover:to-indigo-500 transition shadow-lg"
            >
              Save Profile Preferences
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
