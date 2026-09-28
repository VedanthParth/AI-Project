import React, { useState, useEffect } from 'react';
import { 
  Compass, 
  Brain, 
  Video, 
  BookOpen, 
  Dumbbell, 
  Sparkles, 
  Settings, 
  RotateCcw, 
  GraduationCap, 
  CheckCircle2, 
  Clock, 
  PlayCircle, 
  ShieldCheck,
  Award
} from 'lucide-react';
import ConceptGraph from './components/ConceptGraph';
import OnboardingModal from './components/OnboardingModal';
import ConsentTrialBanner from './components/ConsentTrialBanner';
import QuizModal from './components/QuizModal';
import AITutorPanel from './components/AITutorPanel';

export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [activeQuizConcept, setActiveQuizConcept] = useState(null);
  const [apiKey, setApiKey] = useState('');

  // Fetch state from Backend Engine
  const fetchState = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/curriculum');
      const curriculumData = await res.json();

      const recRes = await fetch('http://localhost:5000/api/recommendation', { method: 'POST' });
      const recData = await recRes.json();

      setData({
        domains: curriculumData.domains,
        concepts: curriculumData.concepts,
        knowledge: recData.knowledge,
        profile: recData.profile,
        recommendation: recData.recommendation
      });
    } catch (err) {
      console.error("Failed to load backend state", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchState();
  }, []);

  const handleSaveProfile = async (newProfile) => {
    try {
      await fetch('http://localhost:5000/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newProfile)
      });
      setShowOnboarding(false);
      fetchState();
    } catch (err) {
      console.error("Error updating profile", err);
    }
  };

  const handleQuizSubmit = async (conceptId, quizResults) => {
    try {
      const answers = quizResults.map(r => ({ questionId: r.questionId, isCorrect: r.isCorrect }));
      const res = await fetch('http://localhost:5000/api/quiz/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conceptId, answers })
      });
      const updatedData = await res.json();
      
      // Update local state with BKT results
      setData(prev => ({
        ...prev,
        knowledge: updatedData.knowledge,
        profile: updatedData.profile,
        recommendation: updatedData.nextRecommendation
      }));
    } catch (err) {
      console.error("Error submitting quiz", err);
    }
  };

  const handleTrialResponse = async (action, targetFormat, durationSessions) => {
    try {
      await fetch('http://localhost:5000/api/trial/respond', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, targetFormat, durationSessions })
      });
      fetchState();
    } catch (err) {
      console.error("Error handling trial response", err);
    }
  };

  const handleResetDemo = async () => {
    await fetch('http://localhost:5000/api/reset', { method: 'POST' });
    fetchState();
  };

  if (loading || !data) {
    return (
      <div className="min-h-screen bg-[#0b0f19] flex items-center justify-center text-cyan-400 font-sans">
        <div className="flex flex-col items-center gap-3">
          <Brain className="w-10 h-10 animate-bounce" />
          <p className="text-sm font-semibold tracking-wider">Initializing Knowledge Tracing Engine...</p>
        </div>
      </div>
    );
  }

  const { recommendation, profile, knowledge, concepts } = data;
  const currentConcept = recommendation?.recommendedConcept;

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 font-sans pb-16 selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 glass-nav px-6 py-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-tr from-cyan-500 to-indigo-600 rounded-xl text-white shadow-lg">
            <GraduationCap className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold gradient-text">PathTrace AI</h1>
            <p className="text-[11px] text-slate-400">Student-Controlled Knowledge-Aware Learning Path Recommender</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowOnboarding(true)}
            className="px-3.5 py-2 rounded-xl glass-card border border-slate-700 text-xs font-semibold text-slate-300 hover:text-white hover:border-cyan-400 transition flex items-center gap-1.5"
          >
            <Settings className="w-4 h-4 text-cyan-400" />
            Edit Preferences
          </button>
          
          <button
            onClick={handleResetDemo}
            className="px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs font-medium text-slate-400 hover:text-rose-400 transition flex items-center gap-1.5"
            title="Reset knowledge tracing state for demo testing"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset State
          </button>
        </div>
      </header>

      {/* Main Container Layout */}
      <main className="max-w-7xl mx-auto px-6 pt-8 space-y-8">
        
        {/* Student Agency Banner / Preferences Summary */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-2xl glass-card border border-slate-800 flex items-center gap-3">
            <div className="p-3 bg-cyan-500/10 text-cyan-400 rounded-xl border border-cyan-500/20">
              {profile.preferredFormat === 'video' ? <Video className="w-5 h-5" /> : profile.preferredFormat === 'text' ? <BookOpen className="w-5 h-5" /> : <Dumbbell className="w-5 h-5" />}
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Resource Preference</div>
              <div className="text-sm font-bold text-white capitalize">{profile.preferredFormat} Format</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl glass-card border border-slate-800 flex items-center gap-3">
            <div className="p-3 bg-indigo-500/10 text-indigo-400 rounded-xl border border-indigo-500/20">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Session Size</div>
              <div className="text-sm font-bold text-white">{profile.sessionLength} Mins / Target</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl glass-card border border-slate-800 flex items-center gap-3">
            <div className="p-3 bg-emerald-500/10 text-emerald-400 rounded-xl border border-emerald-500/20">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Preference Control</div>
              <div className="text-sm font-bold text-emerald-400">Stable (Locked)</div>
            </div>
          </div>

          <div className="p-4 rounded-2xl glass-card border border-slate-800 flex items-center gap-3">
            <div className="p-3 bg-amber-500/10 text-amber-400 rounded-xl border border-amber-500/20">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Active Trial</div>
              <div className="text-sm font-bold text-amber-300">
                {profile.trialActive ? `${profile.activeTrialFormat} (${profile.trialRemaining} remaining)` : "None (User Default)"}
              </div>
            </div>
          </div>
        </div>

        {/* Consent-Based Exploration Trial Proposal Banner */}
        <ConsentTrialBanner 
          proposal={recommendation.trialProposal} 
          onRespond={handleTrialResponse} 
        />

        {/* Next Topic Recommendation Card */}
        {currentConcept && (
          <div className="p-6 rounded-2xl glass-card border border-cyan-500/40 glow-cyan relative overflow-hidden">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
              <div className="space-y-2 max-w-2xl">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 text-[10px] font-bold bg-cyan-500/20 text-cyan-300 rounded-full border border-cyan-400/40 uppercase tracking-wider animate-pulse">
                    Next Recommended Concept
                  </span>
                  <span className="text-xs text-slate-400 font-mono">
                    BKT Mastery: {(recommendation.currentMastery * 100).toFixed(0)}%
                  </span>
                </div>

                <h2 className="text-2xl font-extrabold text-white">{currentConcept.title}</h2>
                <p className="text-xs text-slate-300 leading-relaxed">{currentConcept.description}</p>

                {/* Resource Format Delivery */}
                <div className="pt-2 flex items-center gap-3">
                  <div className="p-2 bg-slate-900 rounded-lg border border-slate-700 text-xs text-cyan-400 flex items-center gap-1.5 font-semibold">
                    {recommendation.recommendedFormat === 'video' ? <Video className="w-4 h-4 text-cyan-400" /> : <BookOpen className="w-4 h-4 text-cyan-400" />}
                    Format: <span className="capitalize text-white">{recommendation.recommendedFormat}</span>
                  </div>
                  <div className="p-2 bg-slate-900 rounded-lg border border-slate-700 text-xs text-indigo-400 flex items-center gap-1.5 font-semibold">
                    <Clock className="w-4 h-4 text-indigo-400" />
                    Target Time: <span className="text-white">{recommendation.sessionLengthMinutes} Mins</span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3 shrink-0">
                <button
                  onClick={() => setActiveQuizConcept(currentConcept)}
                  className="px-6 py-3 text-xs font-bold text-slate-950 bg-gradient-to-r from-cyan-400 via-teal-400 to-emerald-400 hover:from-cyan-300 hover:to-emerald-300 rounded-xl transition shadow-xl flex items-center justify-center gap-2"
                >
                  <PlayCircle className="w-5 h-5" /> Start Quiz Session
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Grid layout for Prerequisite DAG Graph + AI Tutor Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <ConceptGraph
              concepts={concepts}
              masteries={knowledge.masteries}
              completedConcepts={knowledge.completedConcepts}
              recommendedId={currentConcept?.id}
              onSelectConcept={(c) => setActiveQuizConcept(c)}
            />
          </div>

          <div>
            {currentConcept && (
              <AITutorPanel
                concept={currentConcept}
                recommendation={recommendation}
                profile={profile}
              />
            )}
          </div>
        </div>

      </main>

      {/* Onboarding Preference Modal */}
      {showOnboarding && (
        <OnboardingModal
          profile={profile}
          onSave={handleSaveProfile}
          onClose={() => setShowOnboarding(false)}
        />
      )}

      {/* Quiz Execution Modal */}
      {activeQuizConcept && (
        <QuizModal
          concept={activeQuizConcept}
          onSubmit={handleQuizSubmit}
          onClose={() => setActiveQuizConcept(null)}
        />
      )}
    </div>
  );
}
