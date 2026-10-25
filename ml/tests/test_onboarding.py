import numpy as np
import pandas as pd
import pytest

from learnpath.config import ModelConfig, TrainConfig
from learnpath.onboarding import (
    BAD,
    GOOD,
    Topics,
    checklist_history,
    curriculum_position,
    marks_from_history,
    simulate,
)
from learnpath.train import train_model


@pytest.fixture(scope="module")
def topics(tiny_processed, tmp_path_factory):
    """Six units in two strands over the synthetic concepts (unit = concept id mod 6)."""
    root = tmp_path_factory.mktemp("topics")
    ucids = tiny_processed.inter.concept_names[1:]
    unit_of = [f"u{i % 6}" for i in range(len(ucids))]
    pd.DataFrame({"ucid": ucids, "level3_id": unit_of}).to_csv(root / "content.csv", index=False)
    rows = [{"kind": "strand", "id": f"s{j}", "parent": "", "stage": "elementary", "exercises": 0, "name_en": f"S{j}"} for j in range(2)]
    rows += [{"kind": "unit", "id": f"u{i}", "parent": f"s{i % 2}", "stage": "elementary", "exercises": 0, "name_en": f"U{i}"} for i in range(6)]
    pd.DataFrame(rows).to_csv(root / "topics.csv", index=False)
    return Topics.load(tiny_processed.inter.concept_names, root / "content.csv", root / "topics.csv")


def test_topics_map_every_concept_to_its_unit(tiny_processed, topics):
    assert len(topics.units) == 6 and len(topics.strands) == 2
    assert (topics.concept_unit[1:] >= 0).all()
    assert sum(len(c) for c in topics.units.concepts) == tiny_processed.inter.num_concepts


def test_curriculum_position_is_a_share_of_the_sequence(tiny_processed):
    position = curriculum_position(tiny_processed.inter, tiny_processed.splits["train"])
    assert position.shape == (tiny_processed.inter.num_concepts + 1,)
    assert ((position >= 0) & (position <= 1)).all()


def test_checklist_history_follows_the_marks_in_curriculum_order(tiny_processed, topics):
    position = curriculum_position(tiny_processed.inter, tiny_processed.splits["train"])
    concepts, correct = checklist_history({0: GOOD, 3: BAD}, topics, tiny_processed.popularity, position, per_unit=2)
    good = set(topics.representatives(0, tiny_processed.popularity, 2).tolist())
    bad = set(topics.representatives(3, tiny_processed.popularity, 2).tolist())
    assert set(concepts.tolist()) == good | bad and len(concepts) == 4 * 3
    assert correct[np.isin(concepts, list(good))].all()
    assert correct[np.isin(concepts, list(bad))].mean() < 0.5
    assert (np.diff(position[concepts]) >= 0).all()
    with pytest.raises(ValueError):
        checklist_history({0: "great"}, topics, tiny_processed.popularity, position)


def test_marks_come_from_the_most_recent_units_and_their_accuracy(topics):
    unit_concepts = [topics.units.concepts[u][0] for u in range(4)]
    concepts = np.array([unit_concepts[0]] * 4 + [unit_concepts[1]] * 4 + [unit_concepts[2]] * 2)
    correct = np.array([1, 1, 1, 1] + [0, 0, 1, 0] + [1, 1])
    assert marks_from_history(concepts, correct, topics, max_units=2, good_accuracy=0.7) == {2: GOOD, 1: BAD}


def test_simulation_scores_questionnaire_and_full_history(tiny_processed, topics, tmp_path):
    cfg = TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu")
    train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "run", log=lambda _: None)
    result = simulate(
        tiny_processed, topics, "test", {"pointer": tmp_path / "run" / "model.pt"}, max_units=3, noise=0.3,
        shortlist_size=10, max_examples=40, n_boot=20, device="cpu", log=lambda _: None,
    )
    assert 0 < result["examples"] <= 40
    assert 0 < result["questions_answered"] <= 3
    assert {"popularity (knows nothing)", "markov, questionnaire", "pointer, full history", "pointer, questionnaire"} <= set(result["methods"])
    for values in result["methods"].values():
        assert 0.0 <= values["ndcg"]["mean"] <= 1.0
