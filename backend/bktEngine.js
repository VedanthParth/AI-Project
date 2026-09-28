// Bayesian Knowledge Tracing (BKT) Model implementation
// standard parameters: P(L0) initial mastery, P(T) transition probability, P(G) guess, P(S) slip

export class BKTModel {
  constructor(params = {}) {
    this.pL0 = params.pL0 || 0.25; // default initial knowledge
    this.pT = params.pT || 0.15;   // probability of learning after opportunity
    this.pG = params.pG || 0.20;   // probability of guessing correctly
    this.pS = params.pS || 0.10;   // probability of slipping (knowing but missing)
  }

  /**
   * Update student mastery state P(L_t) given quiz accuracy
   * @param {number} currentP Mastery probability before response (0 to 1)
   * @param {boolean} isCorrect Whether response was correct
   * @returns {number} Updated mastery probability P(L_{t+1})
   */
  updateMastery(currentP, isCorrect) {
    let pL = currentP !== undefined && currentP !== null ? currentP : this.pL0;
    
    // Posterior state computation P(L_t | Answer)
    let pL_given_obs;
    if (isCorrect) {
      const pObsGivenL = 1 - this.pS;
      const pObsGivenNotL = this.pG;
      const pObs = (pL * pObsGivenL) + ((1 - pL) * pObsGivenNotL);
      pL_given_obs = (pL * pObsGivenL) / (pObs || 0.0001);
    } else {
      const pObsGivenL = this.pS;
      const pObsGivenNotL = 1 - this.pG;
      const pObs = (pL * pObsGivenL) + ((1 - pL) * pObsGivenNotL);
      pL_given_obs = (pL * pObsGivenL) / (pObs || 0.0001);
    }

    // Transition state update P(L_{t+1})
    let nextP = pL_given_obs + (1 - pL_given_obs) * this.pT;
    
    // Clamp between 0.01 and 0.99
    return Math.max(0.01, Math.min(0.99, Number(nextP.toFixed(4))));
  }
}

/**
   * Rule-Based Recommendation & Prerequisite Graph Solver
   */
export function generateRecommendations(knowledgeState, studentPreferences, curriculum) {
  const { masteries, completedConcepts = [] } = knowledgeState;
  const { preferredFormat, sessionLength, learningStyle, trialActive, activeTrialFormat } = studentPreferences;

  // 1. Identify Eligible Concepts whose prerequisites are fulfilled
  const eligibleConcepts = curriculum.filter(concept => {
    // Skip if already mastered / completed (> 0.85 mastery or in completed list)
    const isMastered = (masteries[concept.id] || 0) >= 0.85 || completedConcepts.includes(concept.id);
    if (isMastered) return false;

    // Check all prerequisites are met (mastery >= 0.70 or completed)
    const prereqsMet = concept.prerequisites.every(prereqId => {
      const prereqMastery = masteries[prereqId] || 0;
      return prereqMastery >= 0.70 || completedConcepts.includes(prereqId);
    });

    return prereqsMet;
  });

  if (eligibleConcepts.length === 0) {
    // All concepts mastered or graph locked
    return {
      recommendedConcept: null,
      message: "Congratulations! You have mastered all available topics in this learning path.",
      eligibleList: [],
      trialProposal: null
    };
  }

  // 2. Score eligible concepts based on BKT Knowledge Gap and Difficulty curve
  // We prioritize concepts with lower mastery (highest potential learning gains)
  const scoredConcepts = eligibleConcepts.map(concept => {
    const currentMastery = masteries[concept.id] || 0.25;
    // Score combines gap (1 - mastery) with logical sequence
    const urgencyScore = (1.0 - currentMastery) * (1.0 - concept.difficulty * 0.3);
    return { concept, currentMastery, urgencyScore };
  });

  scoredConcepts.sort((a, b) => b.urgencyScore - a.urgencyScore);

  const selectedTarget = scoredConcepts[0].concept;
  const selectedMastery = masteries[selectedTarget.id] || 0.25;

  // 3. Determine Resource Type based on STABLE User Preferences (or active user-accepted trial)
  let effectiveFormat = preferredFormat; // e.g. "video", "text", "practice"
  if (trialActive && activeTrialFormat) {
    effectiveFormat = activeTrialFormat;
  }

  // 4. Consent-Based Exploration Proposer Rule:
  // If student has taken >= 3 practice sessions in their current format with stagnant mastery (< 0.50),
  // OR if student has "practice" format but high quiz slip rate, suggest a subtle 3-session trial alternative!
  let trialProposal = null;
  if (!trialActive) {
    if (effectiveFormat === "video" && selectedMastery < 0.45) {
      trialProposal = {
        id: `trial_${Date.now()}`,
        targetFormat: "practice",
        reasoning: `We noticed concept "${selectedTarget.title}" benefits from hands-on problem solving. Would you like to try a 3-session Practice-First trial?`,
        durationSessions: 3
      };
    } else if (effectiveFormat === "text" && selectedMastery < 0.40) {
      trialProposal = {
        id: `trial_${Date.now()}`,
        targetFormat: "video",
        reasoning: `Visual demonstrations can boost comprehension for complex matrix/calculus topics. Would you like to try a 3-session Video trial?`,
        durationSessions: 3
      };
    } else if (effectiveFormat === "practice" && selectedMastery < 0.40) {
      trialProposal = {
        id: `trial_${Date.now()}`,
        targetFormat: "text",
        reasoning: `Reading structured explanations before coding can clarify foundational theorems. Would you like to try a 3-session Explanation-First trial?`,
        durationSessions: 3
      };
    }
  }

  return {
    recommendedConcept: selectedTarget,
    currentMastery: selectedMastery,
    recommendedFormat: effectiveFormat,
    sessionLengthMinutes: sessionLength,
    learningStyle: learningStyle,
    trialProposal: trialProposal,
    allEligible: scoredConcepts.map(s => ({ id: s.concept.id, title: s.concept.title, score: s.urgencyScore.toFixed(2) }))
  };
}
