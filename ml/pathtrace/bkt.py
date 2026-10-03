"""Bayesian Knowledge Tracing: per-concept fit, mastery replay, path reward.

Each concept c has four parameters: p_init (mastered before any practice),
p_learn (an attempt moves an unmastered student to mastered), p_guess
(correct while unmastered) and p_slip (wrong while mastered). There is no
forgetting.

Fitting. EM (Baum-Welch) on every training student's attempts at each
concept, treated as an independent two-state hidden Markov chain. All chains
are processed together in a packed, time-major layout (like a PyTorch
PackedSequence), so one EM iteration is a single forward and backward sweep.
Guess and slip are capped (0.3 by default), which keeps the model
identifiable: a correct answer always raises the mastery estimate. Concepts
with too few training attempts use one global parameter set fitted on all
chains.

Reward. Observing an answer leaves expected mastery unchanged (the posterior
is a martingale), so the only expected change from one practice attempt is
the learning step:

    E[gain_c] = (1 - m_c) * p_learn_c

where m_c is the student's current mastery of c. A path's reward is the sum
over its concepts. That sum is additive, so the best path under BKT is just
the K candidates with the largest expected gain (``oracle_paths``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathtrace import metrics
from pathtrace.preprocess import Examples, Interactions, Processed
from pathtrace.utils import save_json

# The single parameter set the paper used for every concept.
PAPER_DEFAULTS = {"p_init": 0.30, "p_learn": 0.15, "p_guess": 0.20, "p_slip": 0.10}
EPS = 1e-4


@dataclass
class BKTParams:
    """Parameters indexed by concept id, shape (C + 1,); index 0 is padding."""

    p_init: np.ndarray
    p_learn: np.ndarray
    p_guess: np.ndarray
    p_slip: np.ndarray
    fitted: np.ndarray  # bool: True = own fit, False = global fallback

    @classmethod
    def constant(cls, num_concepts: int, p_init: float, p_learn: float, p_guess: float, p_slip: float) -> "BKTParams":
        def full(value: float) -> np.ndarray:
            return np.full(num_concepts + 1, value, dtype=np.float64)

        return cls(full(p_init), full(p_learn), full(p_guess), full(p_slip), np.zeros(num_concepts + 1, dtype=bool))

    @classmethod
    def paper_defaults(cls, num_concepts: int) -> "BKTParams":
        return cls.constant(num_concepts, **PAPER_DEFAULTS)

    def save(self, path: str | Path) -> None:
        np.savez(path, p_init=self.p_init, p_learn=self.p_learn, p_guess=self.p_guess, p_slip=self.p_slip, fitted=self.fitted)

    @classmethod
    def load(cls, path: str | Path) -> "BKTParams":
        with np.load(path) as z:
            return cls(z["p_init"], z["p_learn"], z["p_guess"], z["p_slip"], z["fitted"])


# --------------------------------------------------------------------------- packed chains


@dataclass
class _Chains:
    """Per-(student, concept) answer chains, packed time-major, longest first.

    Step t holds the first ``n_active[t]`` chains in rows
    ``offsets[t]:offsets[t] + n_active[t]`` of the flat arrays.
    """

    y: np.ndarray  # (total,) 0/1 answers
    concept: np.ndarray  # (total,) concept of each row
    chain_concept: np.ndarray  # (n_chains,) concept of each chain, longest first
    n_active: np.ndarray  # (T,)
    offsets: np.ndarray  # (T + 1,)

    def rows(self, t: int, n: int | None = None) -> slice:
        start = self.offsets[t]
        return slice(start, start + (self.n_active[t] if n is None else n))


def _build_chains(inter: Interactions, students: np.ndarray, max_len: int) -> _Chains:
    owner = np.repeat(np.arange(inter.num_students), np.diff(inter.offsets))
    rows = np.flatnonzero(np.isin(owner, students))
    if rows.size == 0:
        raise ValueError("no attempts for the given students")
    concept, own, y = inter.concept[rows], owner[rows], inter.correct[rows]
    # Group by (concept, student); within a group, rows keep time order.
    order = np.lexsort((rows, own, concept))
    concept, own, y = concept[order], own[order], y[order]
    starts_mask = np.r_[True, (concept[1:] != concept[:-1]) | (own[1:] != own[:-1])]
    chain = np.cumsum(starts_mask) - 1
    starts = np.flatnonzero(starts_mask)
    pos = np.arange(len(concept)) - starts[chain]
    keep = pos < max_len
    chain, pos, y = chain[keep], pos[keep], y[keep]
    chain_concept = concept[starts]
    lengths = np.bincount(chain, minlength=len(starts))

    by_length = np.argsort(-lengths, kind="stable")
    rank = np.empty_like(by_length)
    rank[by_length] = np.arange(len(by_length))
    sorted_lengths = lengths[by_length]
    steps = int(sorted_lengths[0])
    n_active = np.searchsorted(-sorted_lengths, -np.arange(steps), side="left")
    offsets = np.r_[0, np.cumsum(n_active)]

    flat = offsets[pos] + rank[chain]
    flat_y = np.empty(len(flat), dtype=np.float64)
    flat_y[flat] = y
    chain_concept = chain_concept[by_length]
    flat_concept = np.concatenate([chain_concept[:n] for n in n_active])
    return _Chains(flat_y, flat_concept, chain_concept, n_active, offsets)


# --------------------------------------------------------------------------- EM


def _forward(ch: _Chains, cid: np.ndarray, p: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filtered P(mastered) and P(unmastered) per row, and each row's P(y_t | y_<t)."""
    total = len(ch.y)
    a_l, a_u, scale = np.empty(total), np.empty(total), np.empty(total)
    init, learn, guess, slip = (p[k][cid] for k in ("p_init", "p_learn", "p_guess", "p_slip"))
    for t in range(len(ch.n_active)):
        n = ch.n_active[t]
        rows = ch.rows(t)
        y = ch.y[rows]
        e_l = np.where(y > 0, 1.0 - slip[:n], slip[:n])
        e_u = np.where(y > 0, guess[:n], 1.0 - guess[:n])
        if t == 0:
            prior = init[:n]
        else:
            prev = ch.rows(t - 1, n)
            prior = a_l[prev] + a_u[prev] * learn[:n]
        joint_l, joint_u = prior * e_l, (1.0 - prior) * e_u
        c = joint_l + joint_u
        a_l[rows], a_u[rows], scale[rows] = joint_l / c, joint_u / c, c
    return a_l, a_u, scale


def _em_step(ch: _Chains, cid: np.ndarray, num: int, p: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    a_l, a_u, scale = _forward(ch, cid, p)
    learn, guess, slip = (p[k][cid] for k in ("p_learn", "p_guess", "p_slip"))
    total = len(ch.y)
    b_l, b_u = np.ones(total), np.ones(total)
    xi = np.zeros(total)  # expected U->L transitions between row t and t+1
    has_next = np.zeros(total, dtype=bool)
    for t in range(len(ch.n_active) - 2, -1, -1):
        n = ch.n_active[t + 1]
        cur, nxt = ch.rows(t, n), ch.rows(t + 1)
        y1 = ch.y[nxt]
        e_l1 = np.where(y1 > 0, 1.0 - slip[:n], slip[:n])
        e_u1 = np.where(y1 > 0, guess[:n], 1.0 - guess[:n])
        c1 = scale[nxt]
        forward_l = e_l1 * b_l[nxt] / c1
        b_l[cur] = forward_l
        b_u[cur] = learn[:n] * forward_l + (1.0 - learn[:n]) * e_u1 * b_u[nxt] / c1
        xi[cur] = a_u[cur] * learn[:n] * forward_l
        has_next[cur] = True
    g_l, g_u = a_l * b_l, a_u * b_u
    cid_rows = ch.concept if num > 1 else np.zeros(total, dtype=np.int64)

    def per_concept(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        if mask is None:
            return np.bincount(cid_rows, weights=values, minlength=num)
        return np.bincount(cid_rows[mask], weights=values[mask], minlength=num)

    starts = np.zeros(total, dtype=bool)
    starts[ch.rows(0)] = True
    n_chains = per_concept(np.ones(total), starts)
    sums = {
        "p_init": (per_concept(g_l, starts), n_chains),
        "p_learn": (per_concept(xi, has_next), per_concept(g_u, has_next)),
        "p_guess": (per_concept(g_u * ch.y), per_concept(g_u)),
        "p_slip": (per_concept(g_l * (1.0 - ch.y)), per_concept(g_l)),
    }
    new = {}
    for key, (num_, den) in sums.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = num_ / den
        new[key] = np.where(den > 0, estimate, p[key])
    return new, float(np.log(scale).sum())


def _run_em(
    ch: _Chains, num: int, init: dict[str, float], max_guess: float, max_slip: float, max_iter: int, tol: float
) -> tuple[dict[str, np.ndarray], list[float]]:
    cid = ch.chain_concept if num > 1 else np.zeros(len(ch.chain_concept), dtype=np.int64)
    p = {k: np.full(num, v, dtype=np.float64) for k, v in init.items()}
    history: list[float] = []
    for _ in range(max_iter):
        new, ll = _em_step(ch, cid, num, p)
        p = {
            "p_init": np.clip(new["p_init"], EPS, 1 - EPS),
            "p_learn": np.clip(new["p_learn"], EPS, 1 - EPS),
            "p_guess": np.clip(new["p_guess"], EPS, max_guess),
            "p_slip": np.clip(new["p_slip"], EPS, max_slip),
        }
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * abs(history[-2]):
            break
    return p, history


def fit_bkt(
    inter: Interactions,
    students: np.ndarray,
    *,
    min_attempts: int = 50,
    max_guess: float = 0.3,
    max_slip: float = 0.3,
    max_chain_len: int = 200,
    max_iter: int = 100,
    tol: float = 1e-5,
    log: Callable[[str], None] | None = None,
) -> tuple[BKTParams, dict]:
    """Fit per-concept BKT on the given (training) students' attempts."""
    ch = _build_chains(inter, students, max_chain_len)
    size = inter.num_concepts + 1
    global_p, global_ll = _run_em(ch, 1, PAPER_DEFAULTS, max_guess, max_slip, max_iter, tol)
    per_p, per_ll = _run_em(ch, size, PAPER_DEFAULTS, max_guess, max_slip, max_iter, tol)
    attempts = np.bincount(ch.concept, minlength=size)
    fitted = attempts >= min_attempts
    fitted[0] = False
    params = BKTParams(
        **{k: np.where(fitted, per_p[k], global_p[k][0]) for k in ("p_init", "p_learn", "p_guess", "p_slip")},
        fitted=fitted,
    )
    info = {
        "attempts": int(len(ch.y)),
        "chains": int(len(ch.chain_concept)),
        "concepts_fitted": int(fitted.sum()),
        "concepts_fallback": int(size - 1 - fitted.sum()),
        "global_params": {k: float(v[0]) for k, v in global_p.items()},
        "em_iterations": {"global": len(global_ll), "per_concept": len(per_ll)},
        "train_loglik_per_attempt": {"global": global_ll[-1] / len(ch.y), "per_concept": per_ll[-1] / len(ch.y)},
    }
    if log:
        log(
            f"BKT: {info['concepts_fitted']} concepts fitted, {info['concepts_fallback']} on the global fallback; "
            f"{len(per_ll)} EM iterations"
        )
    return params, info


def log_likelihood(inter: Interactions, students: np.ndarray, params: BKTParams, max_chain_len: int = 200) -> float:
    """Mean log-likelihood per attempt of the given students' answers."""
    ch = _build_chains(inter, students, max_chain_len)
    p = {"p_init": params.p_init, "p_learn": params.p_learn, "p_guess": params.p_guess, "p_slip": params.p_slip}
    _, _, scale = _forward(ch, ch.chain_concept, p)
    return float(np.log(scale).mean())


# --------------------------------------------------------------------------- simulation


def update(m: float, correct: bool, p_learn: float, p_guess: float, p_slip: float) -> float:
    """One BKT step: condition on the answer, then apply the learning transition."""
    if correct:
        post = m * (1.0 - p_slip) / (m * (1.0 - p_slip) + (1.0 - m) * p_guess)
    else:
        post = m * p_slip / (m * p_slip + (1.0 - m) * (1.0 - p_guess))
    return post + (1.0 - post) * p_learn


def candidate_mastery(inter: Interactions, ex: Examples, params: BKTParams) -> np.ndarray:
    """Mastery of each example's candidates after replaying its history, shape (E, N)."""
    out = np.empty(ex.candidates.shape, dtype=np.float64)
    learn, guess, slip = params.p_learn.tolist(), params.p_guess.tolist(), params.p_slip.tolist()
    init = params.p_init.tolist()
    current, mastery, pos, seq, corr = -1, [], 0, [], []
    for i in np.lexsort((ex.cut, ex.student)):
        student = int(ex.student[i])
        if student != current:
            current, mastery, pos = student, list(init), 0
            concepts, correct = inter.sequence(student)
            seq, corr = concepts.tolist(), correct.tolist()
        for t in range(pos, int(ex.cut[i])):
            c = seq[t]
            mastery[c] = update(mastery[c], corr[t] > 0, learn[c], guess[c], slip[c])
        pos = max(pos, int(ex.cut[i]))
        out[i] = [mastery[c] for c in ex.candidates[i]]
    return out


def expected_gain(mastery: np.ndarray, candidates: np.ndarray, params: BKTParams) -> np.ndarray:
    """Expected one-attempt mastery gain of each candidate, shape (E, N)."""
    return (1.0 - mastery) * params.p_learn[candidates]


def path_reward(gains: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Expected BKT reward of each path of candidate slots, shape (E,)."""
    return np.take_along_axis(gains, paths, axis=1).sum(axis=1)


def oracle_paths(gains: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """The reward-maximising K-path: the K largest expected gains."""
    return metrics.top_k(gains, k, np.random.default_rng(seed))


# --------------------------------------------------------------------------- command


def fit_and_report(proc: Processed, out_dir: str | Path, log: Callable[[str], None] = print, **fit_kwargs) -> dict:
    """Fit on the train split, save parameters, and summarise fit quality and reward spread."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    params, info = fit_bkt(proc.inter, proc.splits["train"], log=log, **fit_kwargs)
    params.save(out / "bkt_params.npz")
    paper = BKTParams.paper_defaults(proc.inter.num_concepts)
    k = proc.data_cfg.path_len

    summary: dict = {"fit": info, "params": {}, "heldout_loglik_per_attempt": {}, "reward": {}}
    for key in ("p_init", "p_learn", "p_guess", "p_slip"):
        values = getattr(params, key)[params.fitted]
        if values.size:
            q = np.percentile(values, [10, 50, 90])
            summary["params"][key] = {"p10": q[0], "median": q[1], "p90": q[2]}
    global_params = BKTParams.constant(proc.inter.num_concepts, **info["global_params"])
    for name, p in (("paper_defaults", paper), ("global_fit", global_params), ("per_concept", params)):
        summary["heldout_loglik_per_attempt"][name] = log_likelihood(proc.inter, proc.splits["val"], p)

    for split in ("val", "test"):
        ex = proc.examples[split]
        if len(ex) == 0:
            continue
        rows = {}
        for name, p in (("paper_defaults", paper), ("per_concept", params)):
            gains = expected_gain(candidate_mastery(proc.inter, ex, p), ex.candidates, p)
            oracle = path_reward(gains, oracle_paths(gains, k))
            rows[name] = {
                "oracle_reward": float(oracle.mean()),
                "random_path_reward": float(gains.mean() * k),
                "target_path_reward": float(path_reward(gains, ex.target_slots).mean()),
                "gain_spread_within_example": float(gains.std(axis=1).mean()),
            }
        summary["reward"][split] = rows
    save_json(summary, out / "bkt_summary.json")
    return summary
