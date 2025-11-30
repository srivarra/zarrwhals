"""Integration tests for DataFrame round-trip through Zarr storage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zarrwhals as zw


class TestBasicRoundTrip:
    """Test basic DataFrame round-trip functionality."""

    def test_simple_numeric_dataframe(self, temp_zarr_store):
        """Test round-trip with simple numeric DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "bool_col": [True, False, True, False, True],
            }
        )

        # Write and read back
        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        # Verify
        pd.testing.assert_frame_equal(result, df)

    def test_string_dataframe(self, temp_zarr_store):
        """Test round-trip with string DataFrame."""
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": ["New York", "San Francisco", "Los Angeles"],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_mixed_types_dataframe(self, temp_zarr_store):
        """Test round-trip with mixed data types."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "value": [1.5, 2.5, 3.5],
                "active": [True, False, True],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_datetime_dataframe(self, temp_zarr_store):
        """Test round-trip with datetime column."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "value": [1, 2, 3, 4, 5],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_categorical_dataframe(self, temp_zarr_store):
        """Test round-trip with categorical column."""
        df = pd.DataFrame(
            {
                "category": pd.Categorical(["A", "B", "A", "C", "B"]),
                "value": [1, 2, 3, 4, 5],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_ordered_categorical(self, temp_zarr_store):
        """Test round-trip with ordered categorical."""
        df = pd.DataFrame(
            {
                "grade": pd.Categorical(
                    ["A", "B", "C", "A", "B"],
                    categories=["C", "B", "A"],
                    ordered=True,
                ),
                "score": [95, 85, 75, 92, 88],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)


class TestIndexHandling:
    """Test DataFrame index preservation."""

    def test_default_index(self, temp_zarr_store):
        """Test with default RangeIndex."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)
        assert result.index.equals(pd.RangeIndex(3))

    def test_named_index(self, temp_zarr_store):
        """Test with named index - now stored positionally."""
        df = pd.DataFrame(
            {"value": [10, 20, 30]},
            index=pd.Index(["a", "b", "c"], name="id"),
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        # Index is now positional (0, 1, 2), not preserved
        expected = pd.DataFrame({"value": [10, 20, 30]})
        pd.testing.assert_frame_equal(result, expected)

    def test_integer_index(self, temp_zarr_store):
        """Test with custom integer index - now stored positionally."""
        df = pd.DataFrame(
            {"value": [10, 20, 30]},
            index=[100, 200, 300],
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        # Index is now positional (0, 1, 2), not preserved
        expected = pd.DataFrame({"value": [10, 20, 30]})
        pd.testing.assert_frame_equal(result, expected)


class TestStorageOptions:
    """Test storage configuration options."""

    def test_mode_w_overwrites_existing(self, temp_zarr_store):
        """Test mode='w' overwrites existing store."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6, 7]})

        # Write initial data
        zw.to_zarr(df1, temp_zarr_store)

        # Overwrite with mode='w' should succeed
        zw.to_zarr(df2, temp_zarr_store, mode="w")

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df2)

    def test_mode_w_minus_fails_if_exists(self, temp_zarr_store):
        """Test that mode='w-' (default) fails when store exists."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        # First write should succeed
        zw.to_zarr(df, temp_zarr_store)

        # Second write with mode='w-' should fail
        with pytest.raises(FileExistsError, match="already exists"):
            zw.to_zarr(df, temp_zarr_store, mode="w-")

    def test_custom_chunks(self, temp_zarr_store):
        """Test with custom chunk size."""
        df = pd.DataFrame({"a": range(100)})

        # Write with small chunks
        zw.to_zarr(df, temp_zarr_store, chunks=10)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_sharding(self, temp_zarr_store):
        """Test with sharding enabled."""
        df = pd.DataFrame(
            {
                "a": list(range(100)),
                "b": [f"item_{i}" for i in range(100)],
            }
        )

        # Write with sharding: 10 rows per chunk, 100 rows per shard
        zw.to_zarr(df, temp_zarr_store, chunks=10, shards=100)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_sharding_large_dataframe(self, temp_zarr_store):
        """Test sharding with larger DataFrame."""
        n = 50000
        df = pd.DataFrame(
            {
                "id": range(n),
                "value": [i * 1.5 for i in range(n)],
                "category": pd.Categorical([f"cat_{i % 10}" for i in range(n)]),
            }
        )

        # 5000 rows per chunk, 50000 rows per shard (10 chunks/shard)
        zw.to_zarr(df, temp_zarr_store, chunks=5000, shards=50000)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_sharding_partial_column_read(self, temp_zarr_store):
        """Test that sharding works with partial column reading."""
        df = pd.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
                "c": range(200, 300),
            }
        )

        zw.to_zarr(df, temp_zarr_store, chunks=10, shards=50)
        result = zw.from_zarr(temp_zarr_store, columns=["a", "c"])

        expected = df[["a", "c"]]
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "compression",
        [
            pytest.param("auto", id="auto"),
            pytest.param(None, id="none"),
        ],
    )
    def test_compression_codecs(self, temp_zarr_store, compression):
        """Test round-trip with different compression settings."""
        df = pd.DataFrame({"a": range(100), "b": [f"item_{i}" for i in range(100)]})
        zw.to_zarr(df, temp_zarr_store, compressors=compression)
        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_compression_with_codec_object(self, temp_zarr_store):
        """Test round-trip with explicit codec objects."""
        from zarr.codecs import ZstdCodec

        df = pd.DataFrame({"a": range(100), "b": [f"item_{i}" for i in range(100)]})
        zw.to_zarr(df, temp_zarr_store, compressors=ZstdCodec(level=5))
        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)


class TestPartialReading:
    """Test reading subsets of DataFrames."""

    def test_select_columns(self, temp_zarr_store):
        """Test reading specific columns only."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store, columns=["a", "c"])

        expected = df[["a", "c"]]
        pd.testing.assert_frame_equal(result, expected)

    def test_invalid_column_selection(self, temp_zarr_store):
        """Test error when requesting non-existent columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        zw.to_zarr(df, temp_zarr_store)

        with pytest.raises(ValueError, match="not found"):
            zw.from_zarr(temp_zarr_store, columns=["a", "nonexistent"])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, temp_zarr_store):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"a": [], "b": []})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_single_row(self, temp_zarr_store):
        """Test with single-row DataFrame."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_single_column(self, temp_zarr_store):
        """Test with single-column DataFrame."""
        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_large_dataframe(self, temp_zarr_store, rng):
        """Test with larger DataFrame (performance check)."""
        # Create a moderately large DataFrame
        n = 10000
        df = pd.DataFrame(
            {
                "int_col": np.arange(n),
                "float_col": rng.random(n),
                "str_col": [f"item_{i}" for i in range(n)],
                "cat_col": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_unicode_strings(self, temp_zarr_store):
        """Test with Unicode strings."""
        df = pd.DataFrame(
            {
                "text": ["Hello", "你好", "Привет", "مرحبا", "🎉"],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_reserved_index_name_error(self, temp_zarr_store):
        """Test that reserved column names are rejected."""
        df = pd.DataFrame(
            {
                "_index": [1, 2, 3],  # Reserved name
                "value": [4, 5, 6],
            }
        )

        with pytest.raises(ValueError, match="reserved name"):
            zw.to_zarr(df, temp_zarr_store)

    def test_duplicate_column_names_error(self, temp_zarr_store):
        """Test that duplicate column names are rejected."""
        # pandas allows duplicate column names, but we don't
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])

        # Narwhals rejects duplicate columns, so we get TypeError
        with pytest.raises(TypeError, match="unique"):
            zw.to_zarr(df, temp_zarr_store)

    def test_nonexistent_store_error(self):
        """Test error when reading non-existent store."""
        with pytest.raises(FileNotFoundError, match="not found"):
            zw.from_zarr("/nonexistent/path.zarr")

    def test_read_nonexistent_column(self, temp_zarr_store):
        """Test error when reading non-existent column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df, temp_zarr_store)

        with pytest.raises(ValueError, match="not found"):
            zw.from_zarr(temp_zarr_store, columns=["a", "nonexistent"])


class TestModeParameter:
    """Test mode parameter behavior."""

    def test_default_mode_is_safe(self, temp_zarr_store):
        """Test default mode='w-' prevents accidental overwrites."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        # First write with default mode should succeed
        zw.to_zarr(df, temp_zarr_store)

        # Second write with default mode should fail
        with pytest.raises(FileExistsError):
            zw.to_zarr(df, temp_zarr_store)

    def test_mode_w_minus_creates_new(self, temp_zarr_store):
        """Test mode='w-' creates new store when doesn't exist."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        zw.to_zarr(df, temp_zarr_store, mode="w-")
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_mode_w_creates_new(self, temp_zarr_store):
        """Test mode='w' creates new store when doesn't exist."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        zw.to_zarr(df, temp_zarr_store, mode="w")
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_mode_w_with_memory_store(self):
        """Test mode='w' with MemoryStore."""
        from zarr.storage import MemoryStore

        df = pd.DataFrame({"a": [1, 2, 3]})
        store = MemoryStore()

        zw.to_zarr(df, store, mode="w")
        result = zw.from_zarr(store)

        pd.testing.assert_frame_equal(result, df)

    def test_mode_w_minus_with_memory_store(self):
        """Test mode='w-' with MemoryStore."""
        from zarr.storage import MemoryStore

        df = pd.DataFrame({"a": [1, 2, 3]})
        store = MemoryStore()

        # First write should succeed
        zw.to_zarr(df, store, mode="w-")
        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df)

        # Second write with mode='w-' should fail
        with pytest.raises(FileExistsError, match="already contains"):
            zw.to_zarr(df, store, mode="w-")

    def test_mode_w_overwrites_memory_store(self):
        """Test mode='w' overwrites existing data in MemoryStore."""
        from zarr.storage import MemoryStore

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6, 7]})
        store = MemoryStore()

        zw.to_zarr(df1, store, mode="w")
        zw.to_zarr(df2, store, mode="w")

        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df2)
