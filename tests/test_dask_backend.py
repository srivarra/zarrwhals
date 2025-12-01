"""Tests for Dask DataFrame backend.

Tests that zarrwhals correctly reads data into Dask DataFrames.
Note: Writing is done with pandas/polars, reading can be done with dask backend.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import zarrwhals as zw

dd = pytest.importorskip("dask.dataframe")


class TestDaskRead:
    """Test reading with Dask backend."""

    def test_dask_basic_read(self, temp_zarr_store: Path):
        """Read stored data as Dask DataFrame."""
        pdf = pd.DataFrame({"a": range(100), "b": [f"item_{i}" for i in range(100)]})
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")

        assert isinstance(result, dd.DataFrame)
        # Check values (dtypes may differ due to arrow string backend)
        computed = result.compute()
        assert list(computed.columns) == list(pdf.columns)
        assert len(computed) == len(pdf)
        pd.testing.assert_series_equal(computed["a"], pdf["a"])
        assert list(computed["b"]) == list(pdf["b"])

    def test_dask_numeric_columns(self, temp_zarr_store: Path):
        """Read numeric columns as Dask DataFrame."""
        pdf = pd.DataFrame(
            {
                "int_col": range(200),
                "float_col": [i * 0.5 for i in range(200)],
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        pd.testing.assert_frame_equal(result.compute(), pdf)

    def test_dask_string_columns(self, temp_zarr_store: Path):
        """Read string columns as Dask DataFrame."""
        pdf = pd.DataFrame(
            {
                "name": [f"person_{i}" for i in range(100)],
                "city": [f"city_{i % 5}" for i in range(100)],
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        computed = result.compute()
        # Check values (string dtype may be PyArrow-backed)
        assert list(computed["name"]) == list(pdf["name"])
        assert list(computed["city"]) == list(pdf["city"])

    def test_dask_datetime_columns(self, temp_zarr_store: Path):
        """Read datetime columns as Dask DataFrame."""
        pdf = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100),
                "value": range(100),
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        pd.testing.assert_frame_equal(result.compute(), pdf)

    def test_dask_categorical_columns(self, temp_zarr_store: Path):
        """Read categorical columns as Dask DataFrame."""
        pdf = pd.DataFrame(
            {
                "category": pd.Categorical([f"cat_{i % 5}" for i in range(100)]),
                "value": range(100),
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        computed = result.compute()
        # Categoricals may not preserve dtype through dask, check values
        assert list(computed["category"]) == list(pdf["category"])
        pd.testing.assert_series_equal(computed["value"], pdf["value"])


class TestDaskPartialReading:
    """Test partial reading with Dask backend."""

    def test_dask_select_columns(self, temp_zarr_store: Path):
        """Read selected columns as Dask DataFrame."""
        pdf = pd.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
                "c": range(200, 300),
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask", columns=["a", "c"])

        expected = pdf[["a", "c"]]
        pd.testing.assert_frame_equal(result.compute(), expected)


class TestDaskWithStorageOptions:
    """Test Dask reading from stores with various options."""

    def test_dask_read_chunked_store(self, temp_zarr_store: Path):
        """Read from store written with explicit chunking."""
        pdf = pd.DataFrame({"data": range(1000)})
        zw.to_zarr(pdf, temp_zarr_store, chunks=100)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        pd.testing.assert_frame_equal(result.compute(), pdf)

    def test_dask_read_sharded_store(self, temp_zarr_store: Path):
        """Read from store written with sharding."""
        pdf = pd.DataFrame({"data": range(1000)})
        zw.to_zarr(pdf, temp_zarr_store, chunks=100, shards=500)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        pd.testing.assert_frame_equal(result.compute(), pdf)


class TestDaskBackendValidation:
    """Test dask backend parameter validation."""

    def test_dask_lazy_false_raises(self, temp_zarr_store: Path):
        """Dask backend with lazy=False should raise ValueError."""
        pdf = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(pdf, temp_zarr_store)

        with pytest.raises(ValueError, match="lazy=False"):
            zw.from_zarr(temp_zarr_store, backend="dask", lazy=False)


@pytest.mark.slow
class TestDaskLargeData:
    """Test Dask with larger datasets."""

    def test_dask_large_dataframe(self, temp_zarr_store: Path, rng):
        """Read large DataFrame with Dask."""
        n = 100_000
        pdf = pd.DataFrame(
            {
                "int_col": rng.integers(0, 1000, n),
                "float_col": rng.random(n),
            }
        )
        zw.to_zarr(pdf, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store, backend="dask")
        pd.testing.assert_frame_equal(result.compute(), pdf)
