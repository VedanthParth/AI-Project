import numpy as np
import pandas as pd

from pathtrace.config import DataConfig
from pathtrace.preprocess import Examples, Interactions, load_log
from pathtrace.prereq import PrereqGraph, example_constraints, infer_graph, path_violations


def _chain_log(tmp_path, num_students=40, length=8, tie_first_two=False):
    """Every student starts c0, c1, ... in order, one 15-minute window each."""
    rows = []
    for s in range(num_students):
        for step in range(length):
            window = 0 if (tie_first_two and step < 2) else step
            stamp = pd.Timestamp("2019-08-01") + pd.Timedelta(minutes=15 * window)
            rows.append((stamp.strftime("%Y-%m-%d %H:%M:%S UTC"), f"s{s:02d}", f"c{step}", "True"))
    path = tmp_path / "chain.csv"
    pd.DataFrame(rows, columns=["timestamp_TW", "uuid", "ucid", "is_correct"]).to_csv(path, index=False)
    return load_log(path, DataConfig())


def _edges(graph, inter):
    names = inter.concept_names
    return {(names[a], names[b]) for a, b in zip(graph.src, graph.dst)}


def test_chain_gives_successor_edges_after_transitive_reduction(tmp_path):
    inter = _chain_log(tmp_path)
    graph, info = infer_graph(inter, np.arange(inter.num_students), min_support=30)
    assert _edges(graph, inter) == {(f"c{i}", f"c{i + 1}") for i in range(7)}
    assert info["edges_before_reduction"] > info["edges"]


def test_same_window_starts_are_not_evidence_of_order(tmp_path):
    inter = _chain_log(tmp_path, tie_first_two=True)
    graph, _ = infer_graph(inter, np.arange(inter.num_students), min_support=30)
    edges = _edges(graph, inter)
    assert ("c0", "c1") not in edges and ("c1", "c0") not in edges
    assert ("c1", "c2") in edges and ("c0", "c2") in edges


def test_inferred_graph_is_acyclic(tiny_processed):
    graph, _ = infer_graph(tiny_processed.inter, tiny_processed.splits["train"], min_support=5, min_proximity=0.0)
    assert len(graph.src) > 0
    indegree = np.bincount(graph.dst, minlength=graph.num_concepts + 1)
    children = {}
    for a, b in zip(graph.src.tolist(), graph.dst.tolist()):
        children.setdefault(a, []).append(b)
    frontier = [c for c in range(graph.num_concepts + 1) if indegree[c] == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in children.get(node, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
    assert visited == graph.num_concepts + 1


def test_constraints_and_violations():
    # Concepts 1..5; prerequisites: 1 -> 3, 2 -> 3, 4 -> 5.
    graph = PrereqGraph(
        src=np.array([1, 2, 4], np.int32),
        dst=np.array([3, 3, 5], np.int32),
        support=np.ones(3),
        precedence=np.ones(3),
        proximity=np.ones(3),
        num_concepts=5,
    )
    inter = Interactions(
        student_uuid=np.array(["s"]),
        concept_names=np.array(["<pad>", "a", "b", "c", "d", "e"]),
        offsets=np.array([0, 2]),
        concept=np.array([1, 1], np.int32),  # the history has concept 1 only
        correct=np.array([1, 1], np.int8),
        time=np.array([0.0, 1.0]),
    )
    ex = Examples(
        student=np.array([0], np.int32),
        cut=np.array([2], np.int32),
        targets=np.array([[3]], np.int32),
        candidates=np.array([[3, 2, 5]], np.int32),  # slots: 0 -> c, 1 -> b, 2 -> e
        target_slots=np.array([[0]]),
        first_group=np.ones(1, np.int8),
    )
    always, need = example_constraints(inter, ex, graph)
    assert always.tolist() == [[False, False, True]]  # e needs d, which is not a candidate
    assert need[0, 0].tolist() == [False, True, False]  # c still needs b (a is in the history)
    assert path_violations(np.array([[1, 0, 2]]), always, need).tolist() == [1]  # b, c ok; e violates
    assert path_violations(np.array([[0, 1, 2]]), always, need).tolist() == [2]  # c before b; e
