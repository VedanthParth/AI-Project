"""An interrupted run, started again, ends exactly where an uninterrupted one does."""

import numpy as np
import pytest
import torch

from learnpath import bkt, prereq
from learnpath.config import ModelConfig, TrainConfig
from learnpath.dkt import DKTConfig, train_dkt
from learnpath.rl import RLConfig, finetune
from learnpath.train import train_model
from learnpath.utils import load_json


class Interrupt(Exception):
    pass


def interrupt_on(text: str):
    """A log callback that stops the run when a line contains ``text``, like a disconnect would."""

    def log(line: str) -> None:
        if text in line:
            raise Interrupt

    return log


def same_weights(a, b) -> bool:
    sa, sb = torch.load(a, weights_only=False)["model_state"], torch.load(b, weights_only=False)["model_state"]
    return all(torch.equal(sa[k], sb[k]) for k in sa)


def test_supervised_training_resumes_mid_run(tiny_processed, tmp_path):
    cfg = TrainConfig(max_epochs=3, patience=10, bootstrap=0, max_history=30, batch_size=32, device="cpu")
    train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "whole", log=lambda _: None)
    with pytest.raises(Interrupt):
        train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "split", log=interrupt_on("epoch   2"))
    assert (tmp_path / "split" / "resume.pt").exists()

    lines: list[str] = []
    train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "split", log=lines.append)
    assert "resuming after epoch 1" in lines
    assert not (tmp_path / "split" / "resume.pt").exists()
    whole, split = load_json(tmp_path / "whole" / "history.json"), load_json(tmp_path / "split" / "history.json")
    assert [r["val_ndcg"] for r in whole] == [r["val_ndcg"] for r in split]
    assert np.array_equal(np.load(tmp_path / "whole" / "preds_test.npy"), np.load(tmp_path / "split" / "preds_test.npy"))
    assert same_weights(tmp_path / "whole" / "model.pt", tmp_path / "split" / "model.pt")


def test_early_stop_survives_a_restart(tiny_processed, tmp_path, monkeypatch):
    import learnpath.train as train_module

    evaluate = train_module.evaluate_model

    def interrupt_final_evaluation(model, proc, split, *args, **kwargs):
        if split == "test":
            raise Interrupt
        return evaluate(model, proc, split, *args, **kwargs)

    # lr=0 never improves, so epoch 2 triggers the early stop; the run is cut off in its final evaluation.
    cfg = TrainConfig(max_epochs=6, patience=1, bootstrap=0, max_history=30, batch_size=32, lr=0.0, device="cpu")
    monkeypatch.setattr(train_module, "evaluate_model", interrupt_final_evaluation)
    with pytest.raises(Interrupt):
        train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "run", log=lambda _: None)
    monkeypatch.undo()
    lines: list[str] = []
    results = train_model(tiny_processed, ModelConfig(), cfg, tmp_path / "run", log=lines.append)
    assert not any(line.startswith("epoch") for line in lines)
    assert results["best_epoch"] == 1


def test_rl_resumes_mid_run(tiny_processed, tmp_path):
    train_model(
        tiny_processed, ModelConfig(), TrainConfig(max_epochs=1, bootstrap=0, max_history=30, batch_size=32, device="cpu"),
        tmp_path / "sup", log=lambda _: None,
    )
    params, _ = bkt.fit_bkt(tiny_processed.inter, tiny_processed.splits["train"], min_attempts=10)
    graph, _ = prereq.infer_graph(tiny_processed.inter, tiny_processed.splits["train"], min_support=5)
    cfg = RLConfig(max_epochs=3, patience=10, lr=2e-3, bootstrap=0, max_history=30, baseline="ema", device="cpu")
    ckpt = tmp_path / "sup" / "model.pt"
    finetune(tiny_processed, ckpt, params, graph, cfg, tmp_path / "whole", log=lambda _: None)
    with pytest.raises(Interrupt):
        finetune(tiny_processed, ckpt, params, graph, cfg, tmp_path / "split", log=interrupt_on("RL epoch  2"))
    lines: list[str] = []
    finetune(tiny_processed, ckpt, params, graph, cfg, tmp_path / "split", log=lines.append)
    assert "resuming after RL epoch 1" in lines
    whole, split = load_json(tmp_path / "whole" / "history.json"), load_json(tmp_path / "split" / "history.json")
    assert [r["val_reward"] for r in whole] == [r["val_reward"] for r in split]
    assert same_weights(tmp_path / "whole" / "model.pt", tmp_path / "split" / "model.pt")


def test_dkt_resumes_mid_run(tiny_processed, tmp_path):
    cfg = DKTConfig(hidden_dim=16, embed_dim=8, max_epochs=3, patience=10, device="cpu")
    whole = train_dkt(tiny_processed, cfg, tmp_path / "whole", log=lambda _: None)
    with pytest.raises(Interrupt):
        train_dkt(tiny_processed, cfg, tmp_path / "split", log=interrupt_on("DKT epoch  2"))
    split = train_dkt(tiny_processed, cfg, tmp_path / "split", log=lambda _: None)
    assert [r["val_auc"] for r in whole["history"]] == [r["val_auc"] for r in split["history"]]
    assert whole["test"]["auc"] == split["test"]["auc"]
