"""Tests for obstore LocalStore backend support."""

import pandas as pd
import pytest
from obstore.store import LocalStore
from zarr.storage import ObjectStore

import zarrwhals as zw


class TestLocalStoreBackend:
    """Test that paths are internally converted to LocalStore."""

    def test_path_uses_localstore(self, tmp_path):
        """Verify path is converted to ObjectStore(LocalStore(...))."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        zw.to_zarr(df, tmp_path / "data.zarr", mode="w")
        result = zw.from_zarr(tmp_path / "data.zarr")
        pd.testing.assert_frame_equal(result, df)

    def test_string_path(self, tmp_path):
        """Test with string path."""
        path = str(tmp_path / "string.zarr")
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, path, mode="w")
        result = zw.from_zarr(path)
        pd.testing.assert_frame_equal(result, df)

    def test_explicit_localstore(self, tmp_path):
        """Test with explicit ObjectStore(LocalStore(...))."""
        store = ObjectStore(LocalStore(prefix=str(tmp_path / "explicit.zarr"), mkdir=True))
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store, mode="w")
        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df)

    def test_mode_w_minus_with_localstore(self, tmp_path):
        """Test mode='w-' fails when store exists with LocalStore."""
        store = ObjectStore(LocalStore(prefix=str(tmp_path / "mode_test.zarr"), mkdir=True))
        df = pd.DataFrame({"a": [1, 2, 3]})
        zw.to_zarr(df, store, mode="w-")

        # Second write with mode='w-' should fail
        with pytest.raises(FileExistsError):
            zw.to_zarr(df, store, mode="w-")

    def test_mode_w_overwrites_with_localstore(self, tmp_path):
        """Test mode='w' overwrites existing data with LocalStore."""
        store = ObjectStore(LocalStore(prefix=str(tmp_path / "overwrite.zarr"), mkdir=True))
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6, 7]})

        zw.to_zarr(df1, store, mode="w")
        zw.to_zarr(df2, store, mode="w")

        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df2)

    def test_with_compression_localstore(self, tmp_path):
        """Test LocalStore with compression."""
        from zarr.codecs import ZstdCodec

        store = ObjectStore(LocalStore(prefix=str(tmp_path / "compressed.zarr"), mkdir=True))
        df = pd.DataFrame({"a": range(100)})
        zw.to_zarr(df, store, mode="w", compressors=ZstdCodec())
        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df)

    def test_with_chunking_localstore(self, tmp_path):
        """Test LocalStore with custom chunking."""
        store = ObjectStore(LocalStore(prefix=str(tmp_path / "chunked.zarr"), mkdir=True))
        df = pd.DataFrame({"a": range(100), "b": range(100)})
        zw.to_zarr(df, store, mode="w", chunks=10)
        result = zw.from_zarr(store)
        pd.testing.assert_frame_equal(result, df)

    def test_polars_backend_with_localstore(self, tmp_path):
        """Test LocalStore with Polars backend."""
        import polars as pl

        store = ObjectStore(LocalStore(prefix=str(tmp_path / "polars.zarr"), mkdir=True))
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        zw.to_zarr(df, store, mode="w")

        result = zw.from_zarr(store, backend="polars", lazy=False)
        expected = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert result.equals(expected)

    def test_get_spec_with_localstore(self, tmp_path):
        """Test get_spec with LocalStore."""
        store = ObjectStore(LocalStore(prefix=str(tmp_path / "spec.zarr"), mkdir=True))
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        zw.to_zarr(df, store, mode="w")

        spec = zw.get_spec(store)
        assert spec.attributes.column_order == ["a", "b"]

    def test_get_spec_with_path(self, tmp_path):
        """Test get_spec with path (internally uses LocalStore)."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        zw.to_zarr(df, tmp_path / "spec_path.zarr", mode="w")

        spec = zw.get_spec(tmp_path / "spec_path.zarr")
        assert spec.attributes.column_order == ["a", "b"]
