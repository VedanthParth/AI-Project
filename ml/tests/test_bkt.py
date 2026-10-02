from itertools import permutations

import numpy as np
import pytest

from pathtrace import bkt
from pathtrace.preprocess import Examples, Interactions

TRUTH = {  # concept id -> (p_init, p_learn, p_guess, p_slip)
    1: (0.20, 0.10, 0.15, 0.08),
    2: (0.45, 0.25, 0.25, 0.12),
    3: (0.10, 0.05, 0.10, 0.05),
}


def _interactions(sequences: list[list[tuple[int, int]]], num_concepts: int) -> Interactions:
    concepts = [c for seq in sequences for c, _ in seq]
    correct = [y for seq in sequences for _, y in seq]
    lengths = [len(seq) for seq in sequences]
    return Interactions(
        student_uuid=np.array([f"s{i}" for i in range(len(sequences))]),
        concept_names=np.array(["<pad>"] + [f"c{i}" for i in range(1, num_concepts + 1)]),
        offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64),
        concept=np.array(concepts, dtype=np.int32),
        correct=np.array(correct, dtype=np.int8),
        time=np.arange(len(concepts), dtype=np.float64),
    )


def _simulate(num_students: int = 1500, chain_len: int = 15, seed: int = 0) -> Interactions:
    """Students practise every concept; attempts at different concepts are interleaved."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(num_students):
        chains = []
        for c, (p_init, p_learn, p_guess, p_slip) in TRUTH.items():
            mastered = rng.random() < p_init
            chain = []
            for _ in range(chain_len):
                p_correct = 1 - p_slip if mastered else p_guess
                chain.append((c, int(rng.random() < p_correct)))
                mastered = mastered or rng.random() < p_learn
            chains.append(chain)
        # Interleave the chains while keeping each chain's own order.
        order = rng.permutation(np.repeat(np.arange(len(chains)), chain_len))
        cursors = [0] * len(chains)
        seq = []
        for k in order:
            seq.append(chains[k][cursors[k]])
            cursors[k] += 1
        sequences.append(seq)
    return _interactions(sequences, num_concepts=len(TRUTH))


def test_em_recovers_known_parameters():
    inter = _simulate()
    params, info = bkt.fit_bkt(inter, np.arange(inter.num_students), min_attempts=10)
    for c, (p_init, p_learn, p_guess, p_slip) in TRUTH.items():
        assert params.fitted[c]
        assert params.p_init[c] == pytest.approx(p_init, abs=0.04), c
        assert params.p_learn[c] == pytest.approx(p_learn, abs=0.03), c
        assert params.p_guess[c] == pytest.approx(p_guess, abs=0.03), c
        assert params.p_slip[c] == pytest.approx(p_slip, abs=0.03), c
    assert info["train_loglik_per_attempt"]["per_concept"] > info["train_loglik_per_attempt"]["global"]


def test_fitted_parameters_beat_paper_defaults_on_heldout_students():
    train, heldout = _simulate(seed=1), _simulate(num_students=300, seed=2)
    params, _ = bkt.fit_bkt(train, np.arange(train.num_students), min_attempts=10)
    students = np.arange(heldout.num_students)
    paper = bkt.BKTParams.paper_defaults(heldout.num_concepts)
    assert bkt.log_likelihood(heldout, students, params) > bkt.log_likelihood(heldout, students, paper)


def test_rare_concepts_fall_back_to_global_parameters():
    rng = np.random.default_rng(3)
    sequences = [[(1, int(rng.random() < 0.7)) for _ in range(10)] for _ in range(30)]
    sequences[0] += [(2, 1), (2, 0)]  # concept 2: only two attempts
    inter = _interactions(sequences, num_concepts=2)
    params, info = bkt.fit_bkt(inter, np.arange(len(sequences)), min_attempts=50)
    assert params.fitted.tolist() == [False, True, False]
    assert params.p_learn[2] == pytest.approx(info["global_params"]["p_learn"])


@pytest.mark.parametrize("m", [0.05, 0.3, 0.7, 0.95])
def test_expected_gain_is_the_learning_step(m):
    learn, guess, slip = 0.2, 0.25, 0.1
    p_correct = m * (1 - slip) + (1 - m) * guess
    expected_next = p_correct * bkt.update(m, True, learn, guess, slip) + (1 - p_correct) * bkt.update(
        m, False, learn, guess, slip
    )
    assert expected_next - m == pytest.approx((1 - m) * learn)


def test_candidate_mastery_replays_the_history():
    inter = _interactions([[(1, 1), (2, 0), (1, 0), (3, 1)]], num_concepts=4)
    params = bkt.BKTParams.paper_defaults(4)
    ex = Examples(
        student=np.array([0, 0], np.int32),
        cut=np.array([3, 1], np.int32),
        targets=np.zeros((2, 1), np.int32),
        candidates=np.array([[1, 2, 3, 4], [1, 2, 3, 4]], np.int32),
        target_slots=np.zeros((2, 1), np.int64),
        first_group=np.ones(2, np.int8),
    )
    m = bkt.candidate_mastery(inter, ex, params)
    p = (0.15, 0.2, 0.1)
    after_c1 = bkt.update(0.3, True, *p)
    expected_cut3 = [bkt.update(after_c1, False, *p), bkt.update(0.3, False, *p), 0.3, 0.3]
    assert m[0].tolist() == pytest.approx(expected_cut3)
    assert m[1].tolist() == pytest.approx([after_c1, 0.3, 0.3, 0.3])


def test_oracle_matches_brute_force():
    rng = np.random.default_rng(4)
    gains = rng.random((50, 7))
    best = np.array([max(sum(row[list(p)]) for p in permutations(range(7), 3)) for row in gains])
    assert bkt.path_reward(gains, bkt.oracle_paths(gains, 3)) == pytest.approx(best)


def test_paper_defaults_give_every_unseen_candidate_the_same_gain():
    params = bkt.BKTParams.paper_defaults(10)
    candidates = np.arange(1, 11).reshape(1, -1)
    gains = bkt.expected_gain(np.full((1, 10), 0.3), candidates, params)
    assert np.allclose(gains, (1 - 0.3) * 0.15)


def test_params_round_trip(tmp_path):
    params = bkt.BKTParams.paper_defaults(5)
    params.save(tmp_path / "bkt.npz")
    loaded = bkt.BKTParams.load(tmp_path / "bkt.npz")
    assert np.array_equal(loaded.p_learn, params.p_learn)
    assert np.array_equal(loaded.fitted, params.fitted)
