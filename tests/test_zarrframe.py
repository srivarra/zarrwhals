"""Tests for ZarrFrame lazy loading behavior."""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from zarrwhals.lazy import ZarrFrame

import zarrwhals as zw


class TestZarrFrameConstruction:
    """Test ZarrFrame construction and metadata access."""

    def test_construction_reads_metadata_only(self, tmp_path):
        """ZarrFrame construction should only read metadata, not data."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        # Construction should succeed
        lzf = ZarrFrame(store_path)

        # Metadata should be available
        assert lzf.columns == ["a", "b"]
        assert "a" in lzf.schema
        assert "b" in lzf.schema

    def test_columns_property(self, tmp_path):
        """columns property should return all column names."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        assert lzf.columns == ["x", "y", "z"]

    def test_nonexistent_store_raises(self, tmp_path):
        """Construction with nonexistent store should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ZarrFrame(tmp_path / "nonexistent.zarr")


class TestLazyColumnSelection:
    """Test lazy column selection."""

    def test_select_returns_new_instance(self, tmp_path):
        """select() should return a new ZarrFrame instance."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        lzf_selected = lzf.select(["a", "b"])

        # Should be different instance
        assert lzf_selected is not lzf

        # Original should be unchanged
        assert lzf.columns == ["a", "b", "c"]

        # New should have selected columns
        assert lzf_selected.columns == ["a", "b"]

    def test_select_invalid_column_raises(self, tmp_path):
        """select() with invalid column should raise ValueError."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)

        with pytest.raises(ValueError, match="not found"):
            lzf.select(["nonexistent"])

    def test_select_partial_invalid_columns_raises(self, tmp_path):
        """select() with mix of valid/invalid columns should raise ValueError."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)

        with pytest.raises(ValueError, match="nonexistent"):
            lzf.select(["a", "nonexistent"])


class TestLazyRowSlicing:
    """Test lazy row slicing."""

    def test_getitem_returns_new_instance(self, tmp_path):
        """__getitem__ with slice should return new ZarrFrame."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": range(100)})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        lzf_sliced = lzf[10:20]

        assert lzf_sliced is not lzf
        # Both should still have same column
        assert lzf_sliced.columns == lzf.columns

    def test_getitem_non_slice_raises(self, tmp_path):
        """__getitem__ with non-slice should raise TypeError."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)

        with pytest.raises(TypeError, match="Only slice"):
            lzf[0]


class TestCollect:
    """Test data materialization via collect()."""

    def test_collect_pandas(self, tmp_path):
        """collect('pandas') should return pandas DataFrame."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.collect("pandas")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert list(result["a"]) == [1, 2, 3]

    def test_collect_polars(self, tmp_path):
        """collect('polars') should return polars DataFrame."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.collect("polars")

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["a", "b"]
        assert result["a"].to_list() == [1, 2, 3]

    def test_collect_with_column_selection(self, tmp_path):
        """collect() after select() should only load selected columns."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.select(["a", "c"]).collect("pandas")

        assert list(result.columns) == ["a", "c"]
        assert "b" not in result.columns

    def test_collect_with_row_slice(self, tmp_path):
        """collect() after row slice should only load sliced rows."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": list(range(100))})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf[10:20].collect("pandas")

        assert len(result) == 10
        assert list(result["a"]) == list(range(10, 20))

    def test_collect_with_selection_and_slice(self, tmp_path):
        """collect() with both column selection and row slice."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame(
            {
                "a": list(range(100)),
                "b": list(range(100, 200)),
                "c": list(range(200, 300)),
            }
        )
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.select(["a", "c"])[25:50].collect("pandas")

        assert list(result.columns) == ["a", "c"]
        assert len(result) == 25
        assert list(result["a"]) == list(range(25, 50))
        assert list(result["c"]) == list(range(225, 250))


class TestNarwhalsProtocol:
    """Test Narwhals protocol compliance."""

    def test_narwhals_lazyframe_protocol(self, tmp_path):
        """ZarrFrame should implement __narwhals_lazyframe__."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)

        assert hasattr(lzf, "__narwhals_lazyframe__")
        assert lzf.__narwhals_lazyframe__() is lzf

    def test_native_namespace_protocol(self, tmp_path):
        """ZarrFrame should implement __native_namespace__."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)

        assert hasattr(lzf, "__native_namespace__")
        assert lzf.__native_namespace__() is zw


class TestPolarsLazy:
    """Test Polars LazyFrame integration."""

    def test_to_polars_lazy_returns_lazyframe(self, tmp_path):
        """to_polars_lazy() should return pl.LazyFrame."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.to_polars_lazy()

        assert isinstance(result, pl.LazyFrame)

    def test_to_polars_lazy_can_collect(self, tmp_path):
        """Polars LazyFrame from to_polars_lazy() should be collectable."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        lazy_result = lzf.to_polars_lazy()
        collected = lazy_result.collect()

        assert isinstance(collected, pl.DataFrame)
        assert collected["a"].to_list() == [1, 2, 3]


class TestCategoricalHandling:
    """Test categorical column handling."""

    def test_categorical_columns(self, tmp_path):
        """ZarrFrame should handle categorical columns."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", "a", "c"]),
                "num": [1, 2, 3, 4],
            }
        )
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf.collect("pandas")

        assert list(result["cat"]) == ["a", "b", "a", "c"]
        assert list(result["num"]) == [1, 2, 3, 4]

    def test_categorical_with_row_slice(self, tmp_path):
        """Categorical columns should work with row slicing."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", "c", "d", "e"] * 20),
                "num": list(range(100)),
            }
        )
        zw.to_zarr(df, store_path)

        lzf = ZarrFrame(store_path)
        result = lzf[10:15].collect("pandas")

        assert len(result) == 5
        assert list(result["num"]) == list(range(10, 15))


class TestIntegrationWithFromZarr:
    """Test that from_zarr() uses ZarrFrame correctly."""

    def test_from_zarr_pandas_unchanged(self, tmp_path):
        """from_zarr() with pandas backend should work as before."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        zw.to_zarr(df, store_path)

        result = zw.from_zarr(store_path, backend="pandas")

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(df, result)

    def test_from_zarr_polars_unchanged(self, tmp_path):
        """from_zarr() with polars backend defaults to lazy."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        zw.to_zarr(df, store_path)

        # Default is lazy=True for polars
        result = zw.from_zarr(store_path, backend="polars")
        assert isinstance(result, pl.LazyFrame)

        # With lazy=False, returns eager DataFrame
        result_eager = zw.from_zarr(store_path, backend="polars", lazy=False)
        assert isinstance(result_eager, pl.DataFrame)
        assert result_eager.columns == ["a", "b"]

    def test_from_zarr_column_projection(self, tmp_path):
        """from_zarr() with columns should use lazy projection."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        zw.to_zarr(df, store_path)

        result = zw.from_zarr(store_path, columns=["a", "c"], backend="pandas")

        assert list(result.columns) == ["a", "c"]
        assert "b" not in result.columns

    def test_from_zarr_polars_lazy(self, tmp_path):
        """from_zarr() with lazy=True should return LazyFrame."""
        store_path = tmp_path / "test.zarr"
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store_path)

        result = zw.from_zarr(store_path, backend="polars", lazy=True)

        assert isinstance(result, pl.LazyFrame)
