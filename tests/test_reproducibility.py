import random

import numpy as np
import pytest
from src.core.reproducibility import set_global_seed


def test_set_global_seed_repeats_python_and_numpy_random_values() -> None:
    set_global_seed(123, deterministic_torch=False)
    first_python_random = random.random()
    first_numpy_random = float(np.random.random())

    set_global_seed(123, deterministic_torch=False)
    second_python_random = random.random()
    second_numpy_random = float(np.random.random())

    assert first_python_random == second_python_random
    assert first_numpy_random == second_numpy_random


def test_set_global_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="Seed must be a non-negative integer"):
        set_global_seed(-1)
