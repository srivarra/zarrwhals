"""Pytest configuration for zarrwhals tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Session-scoped random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def temp_zarr_store():
    """Create a temporary directory for Zarr stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.zarr"


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Standard test DataFrame with mixed types."""
    return pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
