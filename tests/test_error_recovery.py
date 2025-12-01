"""Tests for error handling and recovery scenarios.

Tests that zarrwhals handles error conditions gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import zarrwhals as zw


class TestReadErrors:
    """Test error handling for read operations."""

    def test_read_nonexistent_store(self, tmp_path: Path):
        """Reading nonexistent store raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            zw.from_zarr(tmp_path / "nonexistent.zarr")

    def test_read_empty_directory(self, tmp_path: Path):
        """Reading empty directory raises appropriate error."""
        empty_dir = tmp_path / "empty.zarr"
        empty_dir.mkdir()

        with pytest.raises((FileNotFoundError, ValueError, KeyError)):
            zw.from_zarr(empty_dir)

    def test_read_corrupted_zarr_json(self, temp_zarr_store: Path):
        """Reading store with corrupted zarr.json fails gracefully."""
        # Write valid store first
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, temp_zarr_store)

        # Corrupt the zarr.json
        zarr_json = temp_zarr_store / "zarr.json"
        zarr_json.write_text("invalid json{{{")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            zw.from_zarr(temp_zarr_store)

    def test_read_missing_column_array(self, temp_zarr_store: Path):
        """Reading store with missing column array fails gracefully."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df, temp_zarr_store)

        # Delete one column's data
        import shutil

        shutil.rmtree(temp_zarr_store / "a")

        with pytest.raises((FileNotFoundError, KeyError, ValueError)):
            zw.from_zarr(temp_zarr_store)

    def test_read_invalid_column_selection(self, temp_zarr_store: Path):
        """Reading with invalid column selection raises error."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df, temp_zarr_store)

        with pytest.raises((KeyError, ValueError)):
            zw.from_zarr(temp_zarr_store, columns=["nonexistent"])


class TestWriteErrors:
    """Test error handling for write operations."""

    def test_write_empty_dataframe_allowed(self, temp_zarr_store: Path):
        """Writing empty DataFrame should work (edge case, not error)."""
        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        zw.to_zarr(df, temp_zarr_store)

        result = zw.from_zarr(temp_zarr_store)
        assert len(result) == 0

    def test_write_mode_w_minus_fails_if_exists(self, temp_zarr_store: Path):
        """mode='w-' should fail if store already exists."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, temp_zarr_store)

        with pytest.raises(FileExistsError):
            zw.to_zarr(df, temp_zarr_store, mode="w-")

    def test_write_overwrites_with_mode_w(self, temp_zarr_store: Path):
        """mode='w' should overwrite existing store."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6, 7]})

        zw.to_zarr(df1, temp_zarr_store)
        zw.to_zarr(df2, temp_zarr_store, mode="w")

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df2)


class TestDataValidationErrors:
    """Test error handling for invalid data."""

    def test_reserved_index_name_error(self, temp_zarr_store: Path):
        """Using reserved '_index' column name raises ValueError."""
        df = pd.DataFrame({"_index": [1, 2, 3], "b": [4, 5, 6]})

        with pytest.raises(ValueError, match="_index"):
            zw.to_zarr(df, temp_zarr_store)

    def test_duplicate_column_names_error(self, temp_zarr_store: Path):
        """Duplicate column names should raise error."""
        # Create DataFrame with duplicate columns using numpy
        import numpy as np

        data = np.array([[1, 2], [3, 4], [5, 6]])
        df = pd.DataFrame(data, columns=["a", "a"])

        # Narwhals raises DuplicateError which is wrapped in TypeError
        with pytest.raises((ValueError, TypeError), match=r"[Dd]uplicate|unique"):
            zw.to_zarr(df, temp_zarr_store)


class TestRecoveryScenarios:
    """Test recovery from partial operations."""

    def test_overwrite_recovers_from_bad_state(self, temp_zarr_store: Path):
        """Overwriting a corrupted store should succeed."""
        # Create initial store
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df1, temp_zarr_store)

        # Corrupt it
        (temp_zarr_store / "zarr.json").write_text("corrupted")

        # Overwrite should work
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        zw.to_zarr(df2, temp_zarr_store, mode="w")

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df2)

    def test_multiple_writes_same_store(self, temp_zarr_store: Path):
        """Multiple sequential writes to same store should work."""
        for i in range(5):
            df = pd.DataFrame({"iteration": [i], "value": [i * 100]})
            zw.to_zarr(df, temp_zarr_store, mode="w")

        result = zw.from_zarr(temp_zarr_store)
        assert result["iteration"].iloc[0] == 4
        assert result["value"].iloc[0] == 400


class TestEdgeCases:
    """Test edge cases that might cause errors."""

    def test_very_long_column_names(self, temp_zarr_store: Path):
        """Very long column names should work."""
        long_name = "a" * 200
        df = pd.DataFrame({long_name: [1, 2, 3]})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_special_characters_in_column_names(self, temp_zarr_store: Path):
        """Column names with special characters should work."""
        df = pd.DataFrame(
            {
                "col with spaces": [1, 2],
                "col.with.dots": [3, 4],
                "col-with-dashes": [5, 6],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_unicode_column_names(self, temp_zarr_store: Path):
        """Unicode column names should work."""
        df = pd.DataFrame(
            {
                "日本語": [1, 2, 3],
                "中文": [4, 5, 6],
                "émojis_🎉": [7, 8, 9],
            }
        )

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_single_row_dataframe(self, temp_zarr_store: Path):
        """Single row DataFrame should round-trip correctly."""
        df = pd.DataFrame({"a": [1], "b": ["single"]})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)

    def test_single_column_dataframe(self, temp_zarr_store: Path):
        """Single column DataFrame should round-trip correctly."""
        df = pd.DataFrame({"only_col": range(100)})

        zw.to_zarr(df, temp_zarr_store)
        result = zw.from_zarr(temp_zarr_store)

        pd.testing.assert_frame_equal(result, df)
