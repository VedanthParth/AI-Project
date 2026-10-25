// Every plain-language name and explanation the app shows, in one place, so the pages agree.
// Technical names appear only in TECHNICAL, which the "For the technically curious" panels use.

export const APP = {
  name: 'Maths Learning Path Recommender',
  tagline: 'Finds the next three maths topics to study, in the right order, and shows why.',
};

// The ways of choosing three topics that the app compares.
export const METHODS = {
  pointer: {
    name: 'Our recommendation',
    short: 'Like students similar to you',
    what: 'Our recommender reads everything we know about the student and picks the three topics that students in a similar place most often went on to, in a sensible order.',
  },
  rl: {
    name: 'Learning-focused version',
    short: 'Focus on learning more',
    what: 'The same recommender after extra practice in a simulated classroom, where it was rewarded for topics that should help more and for not skipping building blocks.',
  },
  markov: {
    name: 'The usual next step',
    short: 'What most students did next',
    what: 'Picks what most students did right after the same recent topics. Simple, but it ignores everything else about the student.',
  },
  popularity: {
    name: 'The most common topics',
    short: 'Most-studied topics overall',
    what: "Picks the topics the most students practised overall. It doesn't look at the student at all.",
  },
  actual: { name: 'What they actually did', short: 'Real next topics', what: 'The topics this student really went on to study next.' },
};

// The axes of the "why this topic" and "how plans compare" spider charts.
export const FACTORS = {
  ready: { label: 'Ready for it', explain: 'How many of its building blocks are already done.' },
  room: { label: 'Room to grow', explain: 'How much of it is probably still to learn.' },
  recent: { label: 'Follows on from recent work', explain: 'How naturally it follows the topics done last.' },
  level: { label: 'Right level now', explain: 'How close it sits in the syllabus to where the student is now.' },
  gain: { label: 'Expected improvement', explain: 'How much practising it is likely to help, compared with the alternatives.' },
  progress: { label: 'Predicted progress', explain: 'How much scores on nearby topics are predicted to rise while practising the plan.' },
  matched: { label: 'Matched what they did', explain: 'How many of the three the student really went on to study next.' },
};
export const TOPIC_AXES = ['ready', 'room', 'recent', 'level', 'gain'];
export const PLAN_AXES = ['ready', 'room', 'recent', 'level', 'gain', 'progress'];
export const EXAMPLE_AXES = ['ready', 'room', 'recent', 'level', 'gain', 'matched'];

// A score between 0 and 1 in words, for tooltips and the table views.
export function inWords(value) {
  if (value === null || value === undefined) return 'not known';
  if (value >= 0.85) return 'very high';
  if (value >= 0.6) return 'high';
  if (value >= 0.4) return 'medium';
  if (value >= 0.15) return 'low';
  return 'very low';
}

export const HOW_IT_WORKS = [
  {
    title: 'Tell us where you are',
    text: 'Three quick questions: your class, how you are doing in each area of maths, and the last exercises you finished.',
  },
  {
    title: 'We work out what you know',
    text: 'Your answers become a short practice history, and from it we estimate how well you understand each topic.',
  },
  {
    title: 'We look at what comes next',
    text: 'Out of all 1,326 exercises, we shortlist the ones students usually take up after work like yours.',
  },
  {
    title: 'We put the best three in order',
    text: 'Our recommender picks three from the shortlist and orders them, so building blocks come first.',
  },
];

// Technical names, for the panels that explain how the app is built.
export const TECHNICAL = [
  ['Recommender', 'A pointer network: a GRU reads the attempt history, a Transformer compares the shortlisted topics, and an LSTM decoder points at one topic per step. Trained on 72,758 Junyi Academy students.'],
  ['Learning-focused version', 'The same network fine-tuned with reinforcement learning (REINFORCE with a greedy baseline) against a BKT reward minus a penalty for skipped prerequisites.'],
  ['Estimated understanding', 'Bayesian Knowledge Tracing (BKT): a per-topic model of the chance a student has mastered it, updated after every answer.'],
  ['Building blocks', 'Prerequisites inferred from the order in which training students first attempted topics.'],
  ['Shortlist and "the usual next step"', 'A Markov model of which topic students start after their last three, fitted on training students.'],
  ['Predicted progress', 'Deep Knowledge Tracing (DKT): an LSTM, never used in training, that imitates a student answering the plan’s questions.'],
];
