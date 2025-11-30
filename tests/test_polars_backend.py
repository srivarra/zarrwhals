"""Tests for polars backend support."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

import zarrwhals as zw


class TestPolarsBackend:
    """Test polars DataFrame support."""

    def test_polars_write_pandas_read(self, temp_zarr_store):
        """Write with polars, read with pandas."""
        df_pl = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
                "c": [1.1, 2.2, 3.3],
            }
        )

        # Write polars DataFrame
        zw.to_zarr(df_pl, temp_zarr_store)

        # Read as pandas
        df_pd = zw.from_zarr(temp_zarr_store, backend="pandas")

        # Verify (convert polars to pandas for comparison)
        expected = df_pl.to_pandas()
        pd.testing.assert_frame_equal(df_pd, expected)

    def test_pandas_write_polars_read(self, temp_zarr_store):
        """Write with pandas, read with polars."""
        df_pd = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
                "c": [1.1, 2.2, 3.3],
            }
        )

        # Write pandas DataFrame
        zw.to_zarr(df_pd, temp_zarr_store)

        # Read as polars (defaults to lazy)
        lf_pl = zw.from_zarr(temp_zarr_store, backend="polars")

        # Verify it's a polars LazyFrame (default)
        assert isinstance(lf_pl, pl.LazyFrame)

        # Collect to DataFrame for comparison
        df_pl = lf_pl.collect()
        assert isinstance(df_pl, pl.DataFrame)

        # Compare values by converting to pandas
        result_pd = df_pl.to_pandas()
        pd.testing.assert_frame_equal(result_pd, df_pd)

    def test_polars_round_trip(self, temp_zarr_store):
        """Write and read back with polars."""
        df_original = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["a", "b", "c", "d", "e"],
                "c": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        # Write and read back (defaults to lazy)
        zw.to_zarr(df_original, temp_zarr_store)
        lf_loaded = zw.from_zarr(temp_zarr_store, backend="polars")

        # Verify it's a polars LazyFrame (default)
        assert isinstance(lf_loaded, pl.LazyFrame)

        # Collect to DataFrame
        df_loaded = lf_loaded.collect()

        # Verify shape and columns match
        assert df_loaded.shape == df_original.shape
        assert list(df_loaded.columns) == list(df_original.columns)

        # Compare via pandas for value comparison
        pd.testing.assert_frame_equal(df_loaded.to_pandas(), df_original.to_pandas())

    def test_polars_categorical(self, temp_zarr_store):
        """Test polars DataFrame with categorical-like data."""
        # Create polars DataFrame - categorical gets converted via pandas
        df_pl = pl.DataFrame(
            {
                "category": ["A", "B", "A", "C", "B"],
                "value": [1, 2, 3, 4, 5],
            }
        )

        zw.to_zarr(df_pl, temp_zarr_store)
        lf_loaded = zw.from_zarr(temp_zarr_store, backend="polars")

        # Verify it's a polars LazyFrame (default)
        assert isinstance(lf_loaded, pl.LazyFrame)

        # Collect and compare via pandas
        df_loaded = lf_loaded.collect()
        pd.testing.assert_frame_equal(df_loaded.to_pandas(), df_pl.to_pandas())

    def test_polars_lazy_default(self, temp_zarr_store):
        """Test that polars backend defaults to lazy=True."""
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df_pd, temp_zarr_store)

        # Default should be lazy
        result = zw.from_zarr(temp_zarr_store, backend="polars")
        assert isinstance(result, pl.LazyFrame)

        # Can collect it
        df = result.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (3, 2)

    def test_polars_lazy_explicit(self, temp_zarr_store):
        """Test explicit lazy=True with polars."""
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df_pd, temp_zarr_store)

        # Explicit lazy=True
        result = zw.from_zarr(temp_zarr_store, backend="polars", lazy=True)
        assert isinstance(result, pl.LazyFrame)

        # Can chain lazy operations
        result_filtered = result.filter(pl.col("a") > 1)
        assert isinstance(result_filtered, pl.LazyFrame)

        # Collect when ready
        df = result_filtered.collect()
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 2)

    def test_polars_eager_explicit(self, temp_zarr_store):
        """Test explicit lazy=False with polars returns DataFrame."""
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df_pd, temp_zarr_store)

        # Explicit lazy=False
        result = zw.from_zarr(temp_zarr_store, backend="polars", lazy=False)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 2)

    def test_pandas_lazy_error(self, temp_zarr_store):
        """Test that lazy=True with pandas raises error."""
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df_pd, temp_zarr_store)

        # Should raise error
        with pytest.raises(ValueError, match="pandas backend does not support lazy=True"):
            zw.from_zarr(temp_zarr_store, backend="pandas", lazy=True)

    def test_pandas_ignores_lazy_false(self, temp_zarr_store):
        """Test that pandas backend ignores lazy=False."""
        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df_pd, temp_zarr_store)

        # Should work fine with lazy=False (ignored)
        result = zw.from_zarr(temp_zarr_store, backend="pandas", lazy=False)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
