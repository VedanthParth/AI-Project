"""Pointer-network path recommender with an auxiliary knowledge-tracing head.

Components (paper Section 4):

- a concept embedding table shared by every part of the model;
- a GRU student encoder over x_t = [e_{c_t}; y_t], whose final state s
  summarises the history;
- a Transformer encoder over the candidate set (no positional encoding: the
  set is unordered), so each candidate is contextualised by the others;
- an LSTM pointer decoder with additive attention that scores only the
  eligible candidates at each step and masks earlier picks;
- a second GRU + MLP that predicts next-attempt correctness (DKT-style),
  trained jointly to shape the shared embeddings.

Decoder step t (h_0 = s, c_0 = 0, prev_0 = learned start vector,
context_0 = mean candidate):

    h_t, c_t   = LSTMCell([prev_{t-1}; context_{t-1}], (h_{t-1}, c_{t-1}))
    score_t,i  = v^T tanh(W_h h_t + W_e e~_i)       for eligible i, else -inf
    alpha_t    = softmax(score_t)
    context_t  = sum_i alpha_t,i e~_i
    a_t        = target (teacher forcing) | argmax (greedy) | sample
    prev_t     = e~_{a_t};   a_t becomes ineligible
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pathtrace.batching import Batch
from pathtrace.config import ModelConfig

NEG_INF = -1e9


@dataclass
class DecodeOutput:
    logits: torch.Tensor  # (B, K, N) masked scores at each step
    actions: torch.Tensor  # (B, K) chosen candidate slots
    log_probs: torch.Tensor  # (B, K) log-probability of each chosen slot
    probs: torch.Tensor  # (B, K, N) attention distribution at each step


class PathRecommender(nn.Module):
    def __init__(self, num_concepts: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, h = cfg.embed_dim, cfg.hidden_dim
        self.embed = nn.Embedding(num_concepts + 1, d, padding_idx=0)
        self.dropout = nn.Dropout(cfg.dropout)

        self.student_gru = nn.GRU(d + 1, h, batch_first=True)

        if cfg.attn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d, cfg.attn_heads, cfg.ff_dim, cfg.dropout, batch_first=True
            )
            self.candidate_encoder: nn.Module = nn.TransformerEncoder(
                layer, cfg.attn_layers, enable_nested_tensor=False
            )
        else:
            self.candidate_encoder = nn.Identity()

        self.start = nn.Parameter(torch.zeros(d))
        self.decoder = nn.LSTMCell(2 * d, h)
        self.attn_h = nn.Linear(h, h, bias=False)
        self.attn_e = nn.Linear(d, h, bias=False)
        self.attn_v = nn.Linear(h, 1, bias=False)

        self.kt_gru = nn.GRU(d + 1, h, batch_first=True)
        self.kt_head = nn.Sequential(nn.Linear(h + d, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1))

    # ------------------------------------------------------------------ encoders

    def _history_inputs(self, batch: Batch) -> torch.Tensor:
        e = self.dropout(self.embed(batch.hist_concepts))
        return torch.cat([e, batch.hist_correct.unsqueeze(-1)], dim=-1)

    def encode_student(self, batch: Batch) -> torch.Tensor:
        packed = pack_padded_sequence(
            self._history_inputs(batch), batch.lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.student_gru(packed)
        return h_n[-1]

    def encode_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        return self.candidate_encoder(self.dropout(self.embed(candidates)))

    # ------------------------------------------------------------------ decoder

    def decode(
        self,
        state: torch.Tensor,
        cand: torch.Tensor,
        steps: int,
        mode: str = "greedy",
        target_slots: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
    ) -> DecodeOutput:
        if mode not in {"teacher", "greedy", "sample"}:
            raise ValueError(f"unknown decode mode {mode!r}")
        if mode == "teacher" and target_slots is None:
            raise ValueError("teacher forcing needs target_slots")
        batch_size, num_cand, _ = cand.shape
        rows = torch.arange(batch_size, device=cand.device)
        keys = self.attn_e(cand)
        mask = torch.ones(batch_size, num_cand, dtype=torch.bool, device=cand.device)
        if eligible is not None:
            mask = mask & eligible
        h, c = state, torch.zeros_like(state)
        prev = self.start.expand(batch_size, -1)
        context = cand.mean(dim=1)

        logits, actions, log_probs, probs = [], [], [], []
        for t in range(steps):
            h, c = self.decoder(torch.cat([prev, context], dim=-1), (h, c))
            scores = self.attn_v(torch.tanh(self.attn_h(h).unsqueeze(1) + keys)).squeeze(-1)
            scores = scores.masked_fill(~mask, NEG_INF)
            logp = F.log_softmax(scores, dim=-1)
            if mode == "teacher":
                action = target_slots[:, t]
            elif mode == "greedy":
                action = scores.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=scores).sample()
            alpha = logp.exp()
            context = torch.bmm(alpha.unsqueeze(1), cand).squeeze(1)
            prev = cand[rows, action]
            mask = mask.clone()
            mask[rows, action] = False

            logits.append(scores)
            actions.append(action)
            log_probs.append(logp.gather(1, action.unsqueeze(1)).squeeze(1))
            probs.append(alpha)
        return DecodeOutput(
            logits=torch.stack(logits, dim=1),
            actions=torch.stack(actions, dim=1),
            log_probs=torch.stack(log_probs, dim=1),
            probs=torch.stack(probs, dim=1),
        )

    def recommend(self, batch: Batch, steps: int, mode: str = "greedy") -> DecodeOutput:
        return self.decode(self.encode_student(batch), self.encode_candidates(batch.candidates), steps, mode)

    # ------------------------------------------------------------------ losses

    def path_loss(self, batch: Batch, steps: int) -> torch.Tensor:
        out = self.decode(
            self.encode_student(batch),
            self.encode_candidates(batch.candidates),
            steps,
            mode="teacher",
            target_slots=batch.target_slots,
        )
        return self.step_loss(out.logits, batch)

    def step_loss(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Teacher-forced cross-entropy averaged over steps, step 1 weighted by ``first_step_weight``.

        With ``first_step_ties`` the step-1 term is -log of the total probability
        of the targets tied for the first time window, so picking any of them
        counts as right, as in the first-step metric. The order among tied
        targets is arbitrary (concept id), so the plain loss asks for one of
        them in particular.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (B, K, N)
        nll = -log_probs.gather(2, batch.target_slots.unsqueeze(-1)).squeeze(-1)  # (B, K)
        if self.cfg.first_step_ties:
            steps = torch.arange(nll.size(1), device=nll.device)
            tied = steps.unsqueeze(0) < batch.first_group.unsqueeze(1)
            first = log_probs[:, 0].gather(1, batch.target_slots).masked_fill(~tied, NEG_INF)
            nll = torch.cat([-torch.logsumexp(first, dim=1, keepdim=True), nll[:, 1:]], dim=1)
        weights = torch.ones(nll.size(1), device=nll.device)
        weights[0] = self.cfg.first_step_weight
        return (nll * weights).sum(dim=1).mean() / weights.sum()

    def kt_logits(self, batch: Batch) -> torch.Tensor:
        """Logit that attempt t+1 is correct given attempts <= t, shape (B, T-1)."""
        x = self._history_inputs(batch)
        packed = pack_padded_sequence(x, batch.lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.kt_gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        next_concept = self.embed(batch.hist_concepts[:, 1:])
        return self.kt_head(torch.cat([out[:, :-1], next_concept], dim=-1)).squeeze(-1)

    @staticmethod
    def kt_mask(batch: Batch, device: torch.device) -> torch.Tensor:
        steps = torch.arange(batch.hist_concepts.size(1) - 1, device=device)
        return steps.unsqueeze(0) < (batch.lengths.to(device) - 1).unsqueeze(1)

    def kt_loss(self, batch: Batch) -> torch.Tensor:
        logits = self.kt_logits(batch)
        mask = self.kt_mask(batch, logits.device).float()
        losses = F.binary_cross_entropy_with_logits(logits, batch.hist_correct[:, 1:], reduction="none")
        return (losses * mask).sum() / mask.sum().clamp(min=1.0)
