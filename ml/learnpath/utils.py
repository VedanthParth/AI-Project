"""Small helpers shared across the pipeline: seeding, devices, JSON I/O, resume points."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_resume(path: str | Path, rng: np.random.Generator, **state: Any) -> None:
    """Save a training loop's state after an epoch, with every random number generator it uses.

    The file is written beside the target and then renamed over it, so an
    interruption mid-write leaves the previous resume point intact.
    """
    path = Path(path)
    state["rng_state"] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "numpy": rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume(path: str | Path, rng: np.random.Generator) -> dict | None:
    """Load a ``save_resume`` file and restore its generators, or return None when there is none.

    A run resumed this way continues exactly as if it had never stopped:
    same shuffles, same dropout masks, same samples.
    """
    path = Path(path)
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved = state.pop("rng_state")
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy_global"])
    rng.bit_generator.state = saved["numpy"]
    torch.set_rng_state(saved["torch"])
    if saved["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["cuda"])
    return state
