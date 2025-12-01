"""Pytest configuration for zarrwhals tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import pytest

import zarrwhals as zw

if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Session-scoped random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def temp_zarr_store(tmp_path: Path) -> Path:
    """Create a temporary Zarr store path (leverages pytest's tmp_path)."""
    return tmp_path / "test.zarr"


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


def assert_zarr_roundtrip(
    df: pd.DataFrame | pl.DataFrame,
    store: Path,
    *,
    write_kwargs: Mapping[str, object] | None = None,
    read_kwargs: Mapping[str, object] | None = None,
) -> None:
    """Assert DataFrame round-trips through Zarr correctly.

    Parameters
    ----------
    df
        Input DataFrame (pandas or polars).
    store
        Path to write/read the Zarr store.
    write_kwargs
        Additional kwargs for zw.to_zarr.
    read_kwargs
        Additional kwargs for zw.from_zarr.

    Raises
    ------
    AssertionError
        If the round-tripped DataFrame doesn't match the original.
    """
    write_kwargs = dict(write_kwargs) if write_kwargs else {}
    read_kwargs = dict(read_kwargs) if read_kwargs else {}

    zw.to_zarr(df, store, **write_kwargs)
    result = zw.from_zarr(store, **read_kwargs)

    if isinstance(df, pl.DataFrame):
        pl.testing.assert_frame_equal(result, df)
    else:
        pd.testing.assert_frame_equal(result, df)
