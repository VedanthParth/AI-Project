import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';
import { CONCEPTS, CURRICULUM_DOMAINS } from './curriculum.js';
import { BKTModel, generateRecommendations } from './bktEngine.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const bkt = new BKTModel();

// In-Memory Storage for Demo Session
let userState = {
  profile: {
    preferredFormat: "video",     // "video" | "text" | "practice"
    sessionLength: 15,            // 10 | 15 | 30 minutes
    learningStyle: "explanation", // "explanation" | "practice_first"
    trialActive: false,
    activeTrialFormat: null,
    trialRemaining: 0
  },
  knowledge: {
    masteries: {
      python_basics: 0.88,        // Pre-unlocked to demonstrate graph progression
      numpy_pandas: 0.30,
      linear_algebra: 0.15,
      gradient_descent: 0.10,
      linear_regression: 0.05,
      neural_networks: 0.05
    },
    completedConcepts: ["python_basics"]
  }
};

// GET /api/curriculum - Get entire concept graph & current masteries
app.get('/api/curriculum', (req, res) => {
  res.json({
    domains: CURRICULUM_DOMAINS,
    concepts: CONCEPTS.ml_fundamentals,
    knowledge: userState.knowledge,
    profile: userState.profile
  });
});

// POST /api/profile - Update rule-based onboarding questionnaire answers
app.post('/api/profile', (req, res) => {
  const { preferredFormat, sessionLength, learningStyle } = req.body;
  
  userState.profile = {
    ...userState.profile,
    preferredFormat: preferredFormat || userState.profile.preferredFormat,
    sessionLength: sessionLength || userState.profile.sessionLength,
    learningStyle: learningStyle || userState.profile.learningStyle
  };

  res.json({
    message: "Study profile updated successfully!",
    profile: userState.profile
  });
});

// POST /api/recommendation - Core recommendation algorithm endpoint
app.post('/api/recommendation', (req, res) => {
  const recommendation = generateRecommendations(
    userState.knowledge,
    userState.profile,
    CONCEPTS.ml_fundamentals
  );

  res.json({
    recommendation,
    profile: userState.profile,
    knowledge: userState.knowledge
  });
});

// POST /api/quiz/submit - Process quiz results with BKT engine
app.post('/api/quiz/submit', (req, res) => {
  const { conceptId, answers } = req.body; 
  // answers: [{ questionId: 'q1', isCorrect: true }, ...]

  const targetConcept = CONCEPTS.ml_fundamentals.find(c => c.id === conceptId);
  if (!targetConcept) {
    return res.status(404).json({ error: "Concept not found" });
  }

  // Calculate score and update BKT step for each question
  let currentP = userState.knowledge.masteries[conceptId] || 0.25;
  let correctCount = 0;

  answers.forEach(ans => {
    if (ans.isCorrect) correctCount++;
    currentP = bkt.updateMastery(currentP, ans.isCorrect);
  });

  userState.knowledge.masteries[conceptId] = currentP;

  // Mark completed if mastery > 0.75
  if (currentP >= 0.75 && !userState.knowledge.completedConcepts.includes(conceptId)) {
    userState.knowledge.completedConcepts.push(conceptId);
  }

  // Progress active trial if running
  if (userState.profile.trialActive) {
    userState.profile.trialRemaining -= 1;
    if (userState.profile.trialRemaining <= 0) {
      userState.profile.trialActive = false;
      userState.profile.activeTrialFormat = null;
    }
  }

  // Get next recommendation after update
  const newRec = generateRecommendations(
    userState.knowledge,
    userState.profile,
    CONCEPTS.ml_fundamentals
  );

  res.json({
    conceptId,
    newMastery: currentP,
    correctCount,
    totalQuestions: answers.length,
    knowledge: userState.knowledge,
    profile: userState.profile,
    nextRecommendation: newRec
  });
});

// POST /api/trial/respond - Handle student consent for trial proposal (Accept/Decline/Postpone)
app.post('/api/trial/respond', (req, res) => {
  const { action, targetFormat, durationSessions } = req.body; // action: "accept" | "decline" | "postpone"

  if (action === "accept") {
    userState.profile.trialActive = true;
    userState.profile.activeTrialFormat = targetFormat;
    userState.profile.trialRemaining = durationSessions || 3;
  }

  res.json({
    message: `Trial response recorded: ${action}`,
    profile: userState.profile
  });
});

// POST /api/ai/explain - Gemini API powered pedagogical explanation & tutor reasoning
app.post('/api/ai/explain', async (req, res) => {
  const { conceptId, userApiKey, promptType } = req.body;
  const apiKey = userApiKey || process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(400).json({ 
      error: "No Gemini API key provided. Please pass your API key in the UI or set GEMINI_API_KEY env variable." 
    });
  }

  const concept = CONCEPTS.ml_fundamentals.find(c => c.id === conceptId);
  const conceptTitle = concept ? concept.title : conceptId;
  const currentMastery = userState.knowledge.masteries[conceptId] || 0.25;

  try {
    const ai = new GoogleGenAI({ apiKey });
    
    let promptText = "";
    if (promptType === "why_recommended") {
      promptText = `You are an AI pedagogical advisor. Explain in 2 concise, encouraging sentences why the student is being recommended the concept "${conceptTitle}" right now. Mention their mastery level (${(currentMastery * 100).toFixed(0)}%), prerequisite completion, and how it aligns with their stable study profile (${userState.profile.preferredFormat} format, ${userState.profile.sessionLength}m session). Do not alter their preferences; reinforce student agency.`;
    } else {
      promptText = `You are an expert AI tutor. Provide a clear, engaging 3-paragraph explanation of "${conceptTitle}" tailored for a student with a ${userState.profile.preferredFormat} preference. Include 1 practical code or math snippet.`;
    }

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: promptText,
    });

    res.json({
      explanation: response.text,
      conceptId
    });
  } catch (err) {
    console.error("Gemini API error:", err);
    res.status(500).json({ error: `Gemini API Error: ${err.message}` });
  }
});

// Reset endpoint for testing
app.post('/api/reset', (req, res) => {
  userState = {
    profile: {
      preferredFormat: "video",
      sessionLength: 15,
      learningStyle: "explanation",
      trialActive: false,
      activeTrialFormat: null,
      trialRemaining: 0
    },
    knowledge: {
      masteries: {
        python_basics: 0.88,
        numpy_pandas: 0.30,
        linear_algebra: 0.15,
        gradient_descent: 0.10,
        linear_regression: 0.05,
        neural_networks: 0.05
      },
      completedConcepts: ["python_basics"]
    }
  };
  res.json({ message: "Demo state reset to initial values", state: userState });
});

app.listen(PORT, () => {
  console.log(`Backend Server running on port ${PORT}`);
});
