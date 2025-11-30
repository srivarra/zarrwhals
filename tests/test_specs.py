"""Tests for Pydantic Zarr specs in Zarrwhals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest
import zarr

import zarrwhals as zw
from zarrwhals.specs import (
    ColumnArrayAttributes,
    DataFrameGroupAttributes,
    DataFrameGroupSpec,
    get_dataframe_spec,
    inspect_dataframe_store,
    validate_dataframe_store,
)


@pytest.fixture
def written_dataframe(temp_zarr_store):
    """Create a sample DataFrame and write it to Zarr for spec testing."""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["a", "b", "c", "d", "e"],
        }
    )
    zw.to_zarr(df, temp_zarr_store)
    return df, temp_zarr_store


class TestDataFrameGroupAttributes:
    """Test DataFrameGroupAttributes model."""

    def test_create_attributes(self):
        """Test creating attributes from dict."""
        attrs_dict = {
            "encoding-type": "dataframe",
            "encoding-version": "0.1.0",
            "column-order": ["a", "b", "c"],
            "_index": "_index",
        }
        attrs = DataFrameGroupAttributes.model_validate(attrs_dict)
        assert attrs.encoding_type == "dataframe"
        assert attrs.encoding_version == "0.1.0"
        assert attrs.column_order == ["a", "b", "c"]
        assert attrs.index_name == "_index"

    def test_attributes_validation(self):
        """Test that invalid attributes raise validation error."""
        with pytest.raises(pydantic.ValidationError, match="Input should be 'dataframe'"):
            DataFrameGroupAttributes.model_validate(
                {
                    "encoding-type": "invalid",
                    "encoding-version": "0.1.0",
                    "column-order": ["a"],
                    "_index": "_index",
                }
            )

    def test_attributes_missing_field(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(pydantic.ValidationError, match="Field required"):
            DataFrameGroupAttributes.model_validate(
                {
                    "encoding-type": "dataframe",
                    "encoding-version": "0.1.0",
                    # Missing column-order and _index
                }
            )


class TestColumnArrayAttributes:
    """Test ColumnArrayAttributes model."""

    def test_create_column_attributes(self):
        """Test creating column attributes."""
        attrs = ColumnArrayAttributes.model_validate(
            {
                "encoding-type": "array",
                "narwhals_dtype": "Int64",
            }
        )
        assert attrs.encoding_type == "array"
        assert attrs.narwhals_dtype == "Int64"  # Stored as string

    def test_column_attributes_optional_narwhals_dtype(self):
        """Test that narwhals_dtype is optional."""
        attrs = ColumnArrayAttributes.model_validate(
            {
                "encoding-type": "array",
            }
        )
        assert attrs.encoding_type == "array"
        assert attrs.narwhals_dtype is None


class TestDataFrameGroupSpec:
    """Test DataFrameGroupSpec loading and validation."""

    def test_from_zarr_valid_store(self, written_dataframe):
        """Test loading spec from valid DataFrame store."""
        _, store_path = written_dataframe
        group = zarr.open_group(store_path, mode="r")
        spec = DataFrameGroupSpec.from_zarr(group)

        assert isinstance(spec, DataFrameGroupSpec)
        assert spec.attributes.encoding_type == "dataframe"
        assert spec.attributes.column_order == ["int_col", "float_col", "str_col"]
        assert spec.attributes.index_name == "_index"

    def test_from_zarr_validates_index_exists(self, temp_zarr_store):
        """Test that spec validation requires _index array."""
        # Create invalid store without _index array
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a"]
        group.attrs["_index"] = "_index"
        # Create column but not index
        arr_a = group.create_array("a", data=np.array([1, 2, 3]))
        arr_a.attrs["encoding-type"] = "array"

        with pytest.raises(ValueError, match=r"Index array.*not found"):
            DataFrameGroupSpec.from_zarr(group)

    def test_from_zarr_validates_all_columns_exist(self, temp_zarr_store):
        """Test that spec validation requires all columns in column-order."""
        # Create invalid store with missing column
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a", "b", "c"]  # c is missing
        group.attrs["_index"] = "_index"
        arr_idx = group.create_array("_index", data=np.array([0, 1, 2]))
        arr_idx.attrs["encoding-type"] = "array"
        arr_a = group.create_array("a", data=np.array([1, 2, 3]))
        arr_a.attrs["encoding-type"] = "array"
        arr_b = group.create_array("b", data=np.array([4, 5, 6]))
        arr_b.attrs["encoding-type"] = "array"
        # Missing "c"

        with pytest.raises(ValueError, match="Columns in column-order not found"):
            DataFrameGroupSpec.from_zarr(group)

    def test_from_zarr_validates_columns_are_arrays(self, temp_zarr_store):
        """Test that spec validation requires columns to be arrays."""
        # Create invalid store with column as group instead of array
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a"]
        group.attrs["_index"] = "_index"
        arr_idx = group.create_array("_index", data=np.array([0, 1, 2]))
        arr_idx.attrs["encoding-type"] = "array"
        group.create_group("a")  # Group instead of array

        with pytest.raises(ValueError, match="must be an array"):
            DataFrameGroupSpec.from_zarr(group)

    def test_from_zarr_validates_1d_arrays(self, temp_zarr_store):
        """Test that spec validation requires 1D arrays."""
        # Create invalid store with 2D array
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a"]
        group.attrs["_index"] = "_index"
        arr_idx = group.create_array("_index", data=np.array([0, 1, 2]))
        arr_idx.attrs["encoding-type"] = "array"
        arr_a = group.create_array("a", data=np.array([[1, 2], [3, 4]]))  # 2D array
        arr_a.attrs["encoding-type"] = "array"

        with pytest.raises(ValueError, match="must be 1D"):
            DataFrameGroupSpec.from_zarr(group)

    def test_spec_members_contain_columns(self, written_dataframe):
        """Test that spec.members contains all columns."""
        _, store_path = written_dataframe
        group = zarr.open_group(store_path, mode="r")
        spec = DataFrameGroupSpec.from_zarr(group)

        assert "_index" in spec.members
        assert "int_col" in spec.members
        assert "float_col" in spec.members
        assert "str_col" in spec.members

    def test_spec_array_shapes(self, written_dataframe):
        """Test that spec contains correct array shapes."""
        _, store_path = written_dataframe
        group = zarr.open_group(store_path, mode="r")
        spec = DataFrameGroupSpec.from_zarr(group)

        index_array = spec.members["_index"]
        assert len(index_array.shape) == 1
        assert index_array.shape[0] == 5

        int_array = spec.members["int_col"]
        assert len(int_array.shape) == 1
        assert int_array.shape[0] == 5


class TestValidateDataFrameStore:
    """Test validate_dataframe_store utility."""

    def test_validate_valid_store(self, written_dataframe):
        """Test validating a valid store."""
        _, store_path = written_dataframe
        spec = validate_dataframe_store(store_path)

        assert isinstance(spec, DataFrameGroupSpec)
        assert spec.attributes.encoding_type == "dataframe"

    def test_validate_invalid_store(self, temp_zarr_store):
        """Test validating an invalid store raises error."""
        # Create invalid store
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a"]
        group.attrs["_index"] = "_index"
        # Missing _index array

        with pytest.raises(ValueError, match=r"Index array.*not found"):
            validate_dataframe_store(temp_zarr_store)

    def test_validate_with_group_object(self, written_dataframe):
        """Test validating with a Zarr group object."""
        _, store_path = written_dataframe
        group = zarr.open_group(store_path, mode="r")
        spec = validate_dataframe_store(group)

        assert isinstance(spec, DataFrameGroupSpec)


class TestGetDataFrameSpec:
    """Test get_dataframe_spec utility."""

    def test_get_spec(self, written_dataframe):
        """Test getting spec from store."""
        _, store_path = written_dataframe
        spec = get_dataframe_spec(store_path)

        assert isinstance(spec, DataFrameGroupSpec)
        assert spec.attributes.encoding_type == "dataframe"


class TestInspectDataFrameStore:
    """Test inspect_dataframe_store utility."""

    def test_inspect_store(self, written_dataframe):
        """Test inspecting a DataFrame store."""
        _, store_path = written_dataframe
        info = inspect_dataframe_store(store_path)

        assert isinstance(info, dict)
        assert "encoding_version" in info
        assert "num_columns" in info
        assert "columns" in info
        assert "index_name" in info
        assert "column_info" in info

        assert info["encoding_version"] == "0.0.1"
        assert info["num_columns"] == 3
        assert info["columns"] == ["int_col", "float_col", "str_col"]
        assert info["index_name"] == "_index"

    def test_inspect_column_info(self, written_dataframe):
        """Test that column_info contains array details."""
        _, store_path = written_dataframe
        info = inspect_dataframe_store(store_path)

        assert "int_col" in info["column_info"]
        col_info = info["column_info"]["int_col"]
        assert "shape" in col_info
        assert "dtype" in col_info
        assert col_info["shape"] == (5,)


class TestIntegrationWithIO:
    """Test integration of specs with IO functions."""

    def test_to_zarr_get_spec(self, temp_zarr_store):
        """Test to_zarr followed by get_spec."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zw.to_zarr(df, temp_zarr_store)
        spec = zw.get_spec(temp_zarr_store)

        assert isinstance(spec, DataFrameGroupSpec)
        assert spec.attributes.column_order == ["a", "b"]

    def test_from_zarr_get_spec(self, written_dataframe):
        """Test from_zarr followed by get_spec."""
        _, store_path = written_dataframe
        df = zw.from_zarr(store_path)
        spec = zw.get_spec(store_path)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(spec, DataFrameGroupSpec)
        assert len(df) == 5
        assert len(spec.attributes.column_order) == 3

    def test_spec_validation_on_read(self, temp_zarr_store):
        """Test that reading validates structure automatically."""
        # Create invalid store
        group = zarr.open_group(temp_zarr_store, mode="w", zarr_format=3)
        group.attrs["encoding-type"] = "dataframe"
        group.attrs["encoding-version"] = "0.1.0"
        group.attrs["column-order"] = ["a"]
        group.attrs["_index"] = "_index"
        # Missing _index array

        # Reading should fail validation
        with pytest.raises(ValueError, match=r"Index array.*not found"):
            zw.from_zarr(temp_zarr_store)
