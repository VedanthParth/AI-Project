"""Synthetic interaction logs with the same columns as Junyi's Log_Problem.csv.

Used for development and tests while the real data is unavailable. Students
work through a hidden curriculum order with noise, return to earlier concepts
for review, and answer correctly with a probability that rises with ability
and practice. That gives popularity, transition and history signals for the
baselines and the model to find. Like Junyi, timestamps are rounded to
15-minute windows and each attempt carries its exercise session and problem
number. Nothing here is a result: numbers produced on synthetic data must
never appear in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_log(
    num_students: int = 1500,
    num_concepts: int = 120,
    mean_length: int = 150,
    seed: int = 0,
    window_minutes: int = 15,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Concept names are shuffled so that id order does not reveal curriculum order.
    names = np.array([f"concept-{i:04d}" for i in rng.permutation(num_concepts)])
    difficulty = rng.normal(0.0, 0.7, num_concepts) + np.linspace(-1.0, 1.0, num_concepts)
    start = pd.Timestamp("2019-08-01")

    times, students, concepts, correct, sessions, problems = [], [], [], [], [], []
    for s in range(num_students):
        uuid = f"student-{rng.integers(16**12):012x}-{s:05d}"
        ability = rng.normal()
        length = int(np.clip(rng.lognormal(np.log(mean_length), 0.6), 5, 20 * mean_length))
        visited = np.zeros(num_concepts, dtype=bool)
        practice = np.zeros(num_concepts)
        recent: list[int] = []
        session_count = np.zeros(num_concepts, dtype=int)
        problem_in_session = 0
        previous = -1
        current = int(rng.integers(0, max(1, num_concepts // 10)))
        clock = float(rng.uniform(0, 3e7))  # seconds after ``start``
        for step in range(length):
            if step == 0 or rng.random() < 0.25:
                unvisited = np.flatnonzero(~visited)
                if unvisited.size == 0:
                    current = int(rng.integers(num_concepts))
                elif rng.random() < 0.8:
                    # Next few unvisited concepts in curriculum order.
                    frontier = unvisited[:3]
                    weights = np.array([0.6, 0.25, 0.15])[: frontier.size]
                    current = int(rng.choice(frontier, p=weights / weights.sum()))
                else:
                    current = int(rng.choice(unvisited))
                visited[current] = True
                recent = (recent + [current])[-5:]
            elif recent and rng.random() < 0.3:
                current = int(rng.choice(recent))
            p_correct = 1.0 / (1.0 + np.exp(-(ability - difficulty[current] + 0.4 * practice[current])))
            practice[current] += 1
            if current != previous:
                session_count[current] += 1
                problem_in_session = 0
            problem_in_session += 1
            previous = current
            clock += float(rng.exponential(90.0))
            times.append(clock)
            students.append(uuid)
            concepts.append(names[current])
            correct.append(bool(rng.random() < p_correct))
            sessions.append(int(session_count[current]))
            problems.append(problem_in_session)

    stamps = start + pd.to_timedelta(np.asarray(times), unit="s")
    if window_minutes:
        stamps = stamps.floor(f"{window_minutes}min")
    df = pd.DataFrame(
        {
            "timestamp_TW": stamps.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "uuid": students,
            "ucid": concepts,
            "upid": rng.integers(0, 10_000, len(students)),
            "problem_number": problems,
            "exercise_problem_repeat_session": sessions,
            "is_correct": correct,
        }
    )
    # Real logs are not grouped by student; make the loader do the sorting.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_log(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_log(**kwargs).to_csv(path, index=False)
    return path
