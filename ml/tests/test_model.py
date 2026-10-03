import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pathtrace.batching import make_batch
from pathtrace.config import ModelConfig
from pathtrace.model import PathRecommender


@pytest.fixture()
def model_and_batch(tiny_processed):
    torch.manual_seed(0)
    model = PathRecommender(tiny_processed.inter.num_concepts, ModelConfig(dropout=0.0))
    ex = tiny_processed.examples["train"]
    batch = make_batch(tiny_processed.inter, ex, np.arange(min(32, len(ex))), max_history=50)
    return model, batch


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_decoder_never_repeats_or_picks_ineligible(model_and_batch, mode):
    model, batch = model_and_batch
    eligible = torch.ones_like(batch.candidates, dtype=torch.bool)
    eligible[:, :4] = False
    out = model.decode(
        model.encode_student(batch), model.encode_candidates(batch.candidates), 5, mode, eligible=eligible
    )
    actions = out.actions.numpy()
    assert all(len(set(row)) == len(row) for row in actions)
    assert (actions >= 4).all()
    assert torch.allclose(out.probs.sum(-1), torch.ones(out.probs.shape[:2]))
    picked = out.probs.gather(2, out.actions.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(picked.log(), out.log_probs, atol=1e-5)


def test_teacher_forcing_scores_targets(model_and_batch):
    model, batch = model_and_batch
    out = model.decode(
        model.encode_student(batch), model.encode_candidates(batch.candidates), 3, "teacher", batch.target_slots
    )
    assert torch.equal(out.actions, batch.target_slots)
    model.cfg = ModelConfig(dropout=0.0, first_step_ties=False)
    expected = F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)), batch.target_slots.reshape(-1))
    assert model.path_loss(batch, 3).item() == pytest.approx(expected.item(), rel=1e-5)


def _step_nll(model, batch):
    out = model.decode(
        model.encode_student(batch), model.encode_candidates(batch.candidates), 3, "teacher", batch.target_slots
    )
    nll = -F.log_softmax(out.logits, -1).gather(2, batch.target_slots.unsqueeze(-1)).squeeze(-1)
    return out.logits, nll


def test_first_step_weight_reweights_step_one(model_and_batch):
    model, batch = model_and_batch
    model.eval()
    model.cfg = ModelConfig(dropout=0.0, first_step_weight=3.0, first_step_ties=False)
    logits, nll = _step_nll(model, batch)
    expected = ((3.0 * nll[:, 0] + nll[:, 1] + nll[:, 2]) / 5.0).mean()
    assert model.step_loss(logits, batch).item() == pytest.approx(expected.item(), rel=1e-5)


def test_tie_aware_first_step_accepts_any_tied_target(model_and_batch):
    model, batch = model_and_batch
    model.eval()
    logits, nll = _step_nll(model, batch)
    batch.first_group = torch.ones_like(batch.first_group)
    model.cfg = ModelConfig(dropout=0.0, first_step_ties=False)
    exact = model.step_loss(logits, batch).item()
    model.cfg = ModelConfig(dropout=0.0, first_step_ties=True)
    assert model.step_loss(logits, batch).item() == pytest.approx(exact, rel=1e-6)  # no ties, same loss

    batch.first_group = torch.full_like(batch.first_group, 2)
    step_one = -torch.logsumexp(F.log_softmax(logits[:, 0], -1).gather(1, batch.target_slots[:, :2]), dim=1)
    expected = ((step_one + nll[:, 1] + nll[:, 2]) / 3.0).mean()
    assert model.step_loss(logits, batch).item() == pytest.approx(expected.item(), rel=1e-5)
    assert model.step_loss(logits, batch).item() < exact


def test_kt_mask_ignores_padding(model_and_batch):
    model, batch = model_and_batch
    mask = model.kt_mask(batch, torch.device("cpu"))
    assert mask.sum().item() == int((batch.lengths - 1).sum())


def test_model_can_fit_a_small_batch(model_and_batch):
    model, batch = model_and_batch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    first = None
    for _ in range(80):
        loss = model.path_loss(batch, 3) + 0.5 * model.kt_loss(batch)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first


def test_ablations_build_and_run(tiny_processed, model_and_batch):
    _, batch = model_and_batch
    model = PathRecommender(tiny_processed.inter.num_concepts, ModelConfig(attn_layers=0, kt_weight=0.0))
    assert model.recommend(batch, 3).actions.shape == (len(batch.index), 3)
