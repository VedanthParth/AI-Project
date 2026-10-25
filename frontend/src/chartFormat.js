// Formatting and constants shared by the charts.
import { PLAN_COLOR } from './api';
import { FACTORS } from './copy';

export { inWords } from './copy';

export const axesOf = (keys) => keys.map((key) => ({ key, ...FACTORS[key] }));

// Series colours for comparisons: the shown recommendation, then the two simple ways of choosing.
export const COMPARE_COLORS = [PLAN_COLOR.pointer, PLAN_COLOR.markov, PLAN_COLOR.rl];

export const pct = (v) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`);
