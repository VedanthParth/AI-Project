from __future__ import annotations

import pytest
import torch

from pathtrace.config import DataConfig
from pathtrace.preprocess import Processed, prepare
from pathtrace.synthetic import write_log

# The test models are tiny; one thread avoids oversubscription when other jobs share the CPU.
torch.set_num_threads(1)


@pytest.fixture(scope="session")
def tiny_log(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "Log_Problem.csv"
    return write_log(path, num_students=200, num_concepts=60, mean_length=50, seed=3)


@pytest.fixture(scope="session")
def tiny_processed(tiny_log, tmp_path_factory) -> Processed:
    out = tmp_path_factory.mktemp("processed")
    return prepare(tiny_log, out, DataConfig(num_candidates=12, max_windows_per_student=8))
