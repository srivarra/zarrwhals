"""Tests for Zarr v3 sharding support."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import zarr

import zarrwhals as zw


@pytest.fixture
def temp_zarr_store():
    """Create a temporary directory for Zarr stores."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.zarr"


class TestSharding:
    """Test sharding functionality."""

    def test_basic_sharding(self, temp_zarr_store):
        """Test DataFrame with sharding enabled."""
        df = pd.DataFrame(
            {
                "a": list(range(100)),
                "b": [f"item_{i}" for i in range(100)],
            }
        )

        # Write with sharding: 10 chunks of size 10, grouped into 1 shard
        zw.to_zarr(df, temp_zarr_store, chunks=10, shards=100)

        # Verify we can read it back
        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_sharding_metadata(self, temp_zarr_store):
        """Verify sharding configuration is stored correctly."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})

        zw.to_zarr(df, temp_zarr_store, chunks=2, shards=4)

        # Open zarr store and verify we can read it back
        root = zarr.open_group(temp_zarr_store, mode="r")

        # Check that arrays exist and have correct shape
        x_array = root["x"]
        assert x_array.shape == (5,)

        # Verify data is correct
        result = zw.from_zarr(temp_zarr_store)
        expected = df
        pd.testing.assert_frame_equal(result, expected)

    def test_sharding_large_dataframe(self, temp_zarr_store):
        """Test sharding with larger DataFrame."""
        # Create a moderately large DataFrame
        n = 50000
        df = pd.DataFrame(
            {
                "id": range(n),
                "value": [i * 1.5 for i in range(n)],
                "category": pd.Categorical([f"cat_{i % 10}" for i in range(n)]),
            }
        )

        # Use sharding: 5000 rows per chunk, 50000 rows per shard (10 chunks/shard)
        zw.to_zarr(df, temp_zarr_store, chunks=5000, shards=50000)

        # Verify round-trip
        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_no_sharding(self, temp_zarr_store):
        """Test that default behavior (no sharding) still works."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Write without sharding (default)
        zw.to_zarr(df, temp_zarr_store)

        # Verify it works
        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_sharding_with_datetime(self, temp_zarr_store):
        """Test sharding with datetime columns."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100),
                "value": range(100),
            }
        )

        zw.to_zarr(df, temp_zarr_store, chunks=20, shards=100)

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_sharding_with_strings(self, temp_zarr_store):
        """Test sharding with string columns."""
        df = pd.DataFrame(
            {
                "name": [f"person_{i}" for i in range(100)],
                "city": [f"city_{i % 5}" for i in range(100)],
            }
        )

        zw.to_zarr(df, temp_zarr_store, chunks=10, shards=50)

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_sharding_with_categorical(self, temp_zarr_store):
        """Test sharding with categorical columns."""
        df = pd.DataFrame(
            {
                "category": pd.Categorical([f"cat_{i % 5}" for i in range(100)]),
                "value": range(100),
            }
        )

        zw.to_zarr(df, temp_zarr_store, chunks=20, shards=100)

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_tuple_sharding_backward_compat(self, temp_zarr_store):
        """Test backward compatibility with tuple specification.

        Although the public API now documents int-only parameters,
        tuples are still accepted internally for backward compatibility.
        """
        df = pd.DataFrame({"a": range(100)})

        # Tuple specification still works (backward compatibility)
        zw.to_zarr(df, temp_zarr_store, chunks=(10,), shards=(100,))

        result = zw.from_zarr(temp_zarr_store)
        pd.testing.assert_frame_equal(result, df)

    def test_sharding_partial_column_read(self, temp_zarr_store):
        """Test that sharding doesn't break partial column reading."""
        df = pd.DataFrame(
            {
                "a": range(100),
                "b": range(100, 200),
                "c": range(200, 300),
            }
        )

        zw.to_zarr(df, temp_zarr_store, chunks=10, shards=50)

        # Read subset
        result = zw.from_zarr(temp_zarr_store, columns=["a", "c"])

        expected = df[["a", "c"]]
        pd.testing.assert_frame_equal(result, expected)
