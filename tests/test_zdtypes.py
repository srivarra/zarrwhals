"""Comprehensive tests for ZDType implementations.

Tests all 14 custom Zarr v3 data types in the zarrwhals.zdtypes module.
Each ZDType is tested for:
- JSON serialization/deserialization (to_json, _from_json_v3)
- Native dtype conversion (to_native_dtype)
- Scalar operations (cast_scalar, default_scalar)
- Zarr v2 rejection
"""

from __future__ import annotations

import numpy as np
import pytest
from zarr.core.dtype import DataTypeValidationError

from zarrwhals.zdtypes import (
    ZNarwhalsArray,
    ZNarwhalsBinary,
    ZNarwhalsCategoricalCategories,
    ZNarwhalsCategoricalCodes,
    ZNarwhalsDate,
    ZNarwhalsDatetime,
    ZNarwhalsDecimal,
    ZNarwhalsDuration,
    ZNarwhalsEnum,
    ZNarwhalsFloat32,
    ZNarwhalsFloat64,
    ZNarwhalsInt8,
    ZNarwhalsInt16,
    ZNarwhalsInt32,
    ZNarwhalsInt64,
    ZNarwhalsList,
    ZNarwhalsObject,
    ZNarwhalsStruct,
    ZNarwhalsTime,
    ZNarwhalsUnknown,
)


class TestZNarwhalsDate:
    """Tests for ZNarwhalsDate dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization for Zarr v3."""
        dtype = ZNarwhalsDate()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.date", "configuration": {}}

    def test_to_json_v2_raises(self):
        """Test that Zarr v2 is not supported."""
        dtype = ZNarwhalsDate()
        with pytest.raises(ValueError, match="only supports Zarr v3"):
            dtype.to_json(zarr_format=2)

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsDate()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsDate._from_json_v3(json_data)
        assert restored == dtype

    def test_from_json_v3_invalid(self):
        """Test that invalid JSON raises error."""
        with pytest.raises(DataTypeValidationError):
            ZNarwhalsDate._from_json_v3({"name": "wrong"})

    def test_to_native_dtype(self):
        """Test conversion to NumPy dtype."""
        dtype = ZNarwhalsDate()
        native = dtype.to_native_dtype()
        assert native == np.dtype("datetime64[D]")

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsDate()
        assert dtype.default_scalar() == np.int32(0)

    def test_cast_scalar_int(self):
        """Test casting integer to date scalar."""
        dtype = ZNarwhalsDate()
        result = dtype.cast_scalar(100)
        assert result == np.int32(100)


class TestZNarwhalsTime:
    """Tests for ZNarwhalsTime dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization with default time unit (ns)."""
        dtype = ZNarwhalsTime()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.time"
        assert result["configuration"]["time_unit"] == "ns"

    def test_to_json_v3_with_time_unit(self):
        """Test JSON serialization with explicit time unit."""
        dtype = ZNarwhalsTime(time_unit="us")
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["time_unit"] == "us"

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsTime(time_unit="ms")
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsTime._from_json_v3(json_data)
        assert restored == dtype
        assert restored.time_unit == "ms"

    def test_to_native_dtype(self):
        """Test conversion to NumPy int64 dtype."""
        dtype = ZNarwhalsTime(time_unit="us")
        native = dtype.to_native_dtype()
        assert native == np.dtype("int64")

    def test_default_scalar(self):
        """Test default scalar value (midnight)."""
        dtype = ZNarwhalsTime()
        assert dtype.default_scalar() == np.int64(0)


class TestZNarwhalsDatetime:
    """Tests for ZNarwhalsDatetime dtype."""

    def test_to_json_v3_default(self):
        """Test JSON serialization with defaults."""
        dtype = ZNarwhalsDatetime()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.datetime"
        assert result["configuration"]["time_unit"] == "ns"
        assert result["configuration"]["time_zone"] is None

    def test_to_json_v3_with_timezone(self):
        """Test JSON serialization with timezone."""
        dtype = ZNarwhalsDatetime(time_unit="us", time_zone="UTC")
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["time_unit"] == "us"
        assert result["configuration"]["time_zone"] == "UTC"

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsDatetime(time_unit="ms", time_zone="America/New_York")
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsDatetime._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy datetime64 dtype."""
        dtype = ZNarwhalsDatetime(time_unit="ns")
        native = dtype.to_native_dtype()
        assert native == np.dtype("datetime64[ns]")

    def test_default_scalar(self):
        """Test default scalar value (epoch)."""
        dtype = ZNarwhalsDatetime()
        assert dtype.default_scalar() == np.int64(0)


class TestZNarwhalsDuration:
    """Tests for ZNarwhalsDuration dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsDuration(time_unit="ns")
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.duration"
        assert result["configuration"]["time_unit"] == "ns"

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsDuration(time_unit="ms")
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsDuration._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy timedelta64 dtype."""
        dtype = ZNarwhalsDuration(time_unit="s")
        native = dtype.to_native_dtype()
        assert native == np.dtype("timedelta64[s]")


class TestZNarwhalsBinary:
    """Tests for ZNarwhalsBinary dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsBinary()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.binary", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsBinary()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsBinary._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy object dtype."""
        dtype = ZNarwhalsBinary()
        native = dtype.to_native_dtype()
        assert native == np.dtype("O")


class TestZNarwhalsDecimal:
    """Tests for ZNarwhalsDecimal dtype."""

    def test_to_json_v3_default(self):
        """Test JSON serialization with defaults."""
        dtype = ZNarwhalsDecimal()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.decimal"
        assert result["configuration"]["precision"] is None
        assert result["configuration"]["scale"] is None

    def test_to_json_v3_with_precision_scale(self):
        """Test JSON serialization with precision and scale."""
        dtype = ZNarwhalsDecimal(precision=10, scale=2)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["precision"] == 10
        assert result["configuration"]["scale"] == 2

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsDecimal(precision=18, scale=6)
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsDecimal._from_json_v3(json_data)
        assert restored == dtype


class TestZNarwhalsList:
    """Tests for ZNarwhalsList dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsList(inner_dtype="Int64")
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.list"
        assert result["configuration"]["inner_dtype"] == "Int64"

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsList(inner_dtype="String")
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsList._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy object dtype."""
        dtype = ZNarwhalsList(inner_dtype="Float64")
        native = dtype.to_native_dtype()
        assert native == np.dtype("O")

    def test_default_scalar(self):
        """Test default scalar value (empty list JSON)."""
        dtype = ZNarwhalsList()
        assert dtype.default_scalar() == "[]"


class TestZNarwhalsArray:
    """Tests for ZNarwhalsArray dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsArray(inner_dtype="Float32", shape=(10,))
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.array"
        assert result["configuration"]["inner_dtype"] == "Float32"
        assert result["configuration"]["shape"] == [10]

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsArray(inner_dtype="Int16", shape=(3, 4))
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsArray._from_json_v3(json_data)
        assert restored.inner_dtype == dtype.inner_dtype
        assert restored.shape == dtype.shape

    def test_to_native_dtype(self):
        """Test conversion to NumPy dtype based on inner_dtype."""
        dtype = ZNarwhalsArray(inner_dtype="Float64", shape=(5,))
        native = dtype.to_native_dtype()
        assert native == np.dtype("float64")

    def test_default_scalar(self):
        """Test default scalar value (zeros array)."""
        dtype = ZNarwhalsArray(inner_dtype="Float32", shape=(3,))
        result = dtype.default_scalar()
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))


class TestZNarwhalsStruct:
    """Tests for ZNarwhalsStruct dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsStruct(fields=(("name", "String"), ("age", "Int32")))
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.struct"
        assert result["configuration"]["fields"] == [
            {"name": "name", "dtype": "String"},
            {"name": "age", "dtype": "Int32"},
        ]

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsStruct(fields=(("x", "Float64"), ("y", "Float64")))
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsStruct._from_json_v3(json_data)
        assert restored.fields == dtype.fields

    def test_to_native_dtype(self):
        """Test conversion to NumPy object dtype."""
        dtype = ZNarwhalsStruct(fields=(("a", "Int64"),))
        native = dtype.to_native_dtype()
        assert native == np.dtype("O")

    def test_default_scalar(self):
        """Test default scalar value (empty object JSON)."""
        dtype = ZNarwhalsStruct()
        assert dtype.default_scalar() == "{}"


class TestZNarwhalsObject:
    """Tests for ZNarwhalsObject dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsObject()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.object", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsObject()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsObject._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy object dtype."""
        dtype = ZNarwhalsObject()
        native = dtype.to_native_dtype()
        assert native == np.dtype("O")


class TestZNarwhalsUnknown:
    """Tests for ZNarwhalsUnknown dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsUnknown()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.unknown", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsUnknown()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsUnknown._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy object dtype."""
        dtype = ZNarwhalsUnknown()
        native = dtype.to_native_dtype()
        assert native == np.dtype("O")


class TestZNarwhalsCategoricalCodes:
    """Tests for ZNarwhalsCategoricalCodes dtype."""

    def test_to_json_v3_default(self):
        """Test JSON serialization with default (unordered)."""
        dtype = ZNarwhalsCategoricalCodes()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.categorical.codes"
        assert result["configuration"]["ordered"] is False

    def test_to_json_v3_ordered(self):
        """Test JSON serialization with ordered=True."""
        dtype = ZNarwhalsCategoricalCodes(ordered=True)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["ordered"] is True

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsCategoricalCodes(ordered=True)
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsCategoricalCodes._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int32 dtype."""
        dtype = ZNarwhalsCategoricalCodes()
        native = dtype.to_native_dtype()
        assert native == np.dtype("int32")

    def test_default_scalar(self):
        """Test default scalar value (-1 for missing)."""
        dtype = ZNarwhalsCategoricalCodes()
        assert dtype.default_scalar() == np.int32(-1)


class TestZNarwhalsCategoricalCategories:
    """Tests for ZNarwhalsCategoricalCategories dtype."""

    def test_to_json_v3_string(self):
        """Test JSON serialization with string inner dtype."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="string")
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.categorical.categories"
        assert result["configuration"]["inner_dtype"] == "string"

    def test_to_json_v3_int64(self):
        """Test JSON serialization with int64 inner dtype."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="int64")
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["inner_dtype"] == "int64"

    def test_to_json_v3_float64(self):
        """Test JSON serialization with float64 inner dtype."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="float64")
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["inner_dtype"] == "float64"

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        for inner in ["string", "int64", "float64"]:
            dtype = ZNarwhalsCategoricalCategories(inner_dtype=inner)
            json_data = dtype.to_json(zarr_format=3)
            restored = ZNarwhalsCategoricalCategories._from_json_v3(json_data)
            assert restored == dtype

    def test_to_native_dtype_string(self):
        """Test conversion to NumPy StringDType for string categories."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="string")
        native = dtype.to_native_dtype()
        assert native == np.dtypes.StringDType()

    def test_to_native_dtype_int64(self):
        """Test conversion to NumPy int64 for int categories."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="int64")
        native = dtype.to_native_dtype()
        assert native == np.dtype("int64")

    def test_to_native_dtype_float64(self):
        """Test conversion to NumPy float64 for float categories."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype="float64")
        native = dtype.to_native_dtype()
        assert native == np.dtype("float64")


class TestZNarwhalsEnum:
    """Tests for ZNarwhalsEnum dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsEnum(categories=("A", "B", "C"))
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.enum"
        assert result["configuration"]["categories"] == ["A", "B", "C"]
        assert result["configuration"]["ordered"] is True

    def test_to_json_v3_unordered(self):
        """Test JSON serialization with ordered=False."""
        dtype = ZNarwhalsEnum(categories=("X", "Y"), ordered=False)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["ordered"] is False

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsEnum(categories=("small", "medium", "large"))
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsEnum._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int32 dtype (codes storage)."""
        dtype = ZNarwhalsEnum(categories=("X", "Y"))
        native = dtype.to_native_dtype()
        assert native == np.dtype("int32")

    def test_default_scalar(self):
        """Test default scalar value (-1 for missing)."""
        dtype = ZNarwhalsEnum(categories=("A", "B"))
        assert dtype.default_scalar() == np.int32(-1)


class TestZNarwhalsFloat32:
    """Tests for ZNarwhalsFloat32 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsFloat32()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.float32", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsFloat32()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsFloat32._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy float32 dtype."""
        dtype = ZNarwhalsFloat32()
        native = dtype.to_native_dtype()
        assert native == np.dtype("float32")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsFloat32()
        assert dtype.cast_scalar(1.5) == np.float32(1.5)
        assert dtype.cast_scalar(42) == np.float32(42.0)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsFloat32()
        assert dtype.default_scalar() == np.float32(0.0)


class TestZNarwhalsFloat64:
    """Tests for ZNarwhalsFloat64 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsFloat64()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.float64", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsFloat64()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsFloat64._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy float64 dtype."""
        dtype = ZNarwhalsFloat64()
        native = dtype.to_native_dtype()
        assert native == np.dtype("float64")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsFloat64()
        assert dtype.cast_scalar(1.5) == np.float64(1.5)
        assert dtype.cast_scalar(42) == np.float64(42.0)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsFloat64()
        assert dtype.default_scalar() == np.float64(0.0)


class TestZNarwhalsInt8:
    """Tests for ZNarwhalsInt8 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsInt8()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.int8", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsInt8()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsInt8._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int8 dtype."""
        dtype = ZNarwhalsInt8()
        native = dtype.to_native_dtype()
        assert native == np.dtype("int8")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsInt8()
        assert dtype.cast_scalar(42) == np.int8(42)
        assert dtype.cast_scalar(-10) == np.int8(-10)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsInt8()
        assert dtype.default_scalar() == np.int8(0)


class TestZNarwhalsInt16:
    """Tests for ZNarwhalsInt16 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsInt16()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.int16", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsInt16()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsInt16._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int16 dtype."""
        dtype = ZNarwhalsInt16()
        native = dtype.to_native_dtype()
        assert native == np.dtype("int16")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsInt16()
        assert dtype.cast_scalar(1000) == np.int16(1000)
        assert dtype.cast_scalar(-500) == np.int16(-500)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsInt16()
        assert dtype.default_scalar() == np.int16(0)


class TestZNarwhalsInt32:
    """Tests for ZNarwhalsInt32 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsInt32()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.int32", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsInt32()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsInt32._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int32 dtype."""
        dtype = ZNarwhalsInt32()
        native = dtype.to_native_dtype()
        assert native == np.dtype("int32")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsInt32()
        assert dtype.cast_scalar(100000) == np.int32(100000)
        assert dtype.cast_scalar(-50000) == np.int32(-50000)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsInt32()
        assert dtype.default_scalar() == np.int32(0)


class TestZNarwhalsInt64:
    """Tests for ZNarwhalsInt64 dtype."""

    def test_to_json_v3(self):
        """Test JSON serialization."""
        dtype = ZNarwhalsInt64()
        result = dtype.to_json(zarr_format=3)
        assert result == {"name": "narwhals.int64", "configuration": {}}

    def test_from_json_v3_roundtrip(self):
        """Test JSON deserialization roundtrip."""
        dtype = ZNarwhalsInt64()
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsInt64._from_json_v3(json_data)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to NumPy int64 dtype."""
        dtype = ZNarwhalsInt64()
        native = dtype.to_native_dtype()
        assert native == np.dtype("int64")

    def test_cast_scalar(self):
        """Test scalar casting."""
        dtype = ZNarwhalsInt64()
        assert dtype.cast_scalar(10000000000) == np.int64(10000000000)
        assert dtype.cast_scalar(-5000000000) == np.int64(-5000000000)

    def test_default_scalar(self):
        """Test default scalar value."""
        dtype = ZNarwhalsInt64()
        assert dtype.default_scalar() == np.int64(0)


class TestZDTypeV2Rejection:
    """Test that all ZDTypes properly reject Zarr v2 format."""

    @pytest.mark.parametrize(
        ("dtype_cls", "kwargs"),
        [
            (ZNarwhalsDate, {}),
            (ZNarwhalsTime, {}),
            (ZNarwhalsDatetime, {}),
            (ZNarwhalsDuration, {}),
            (ZNarwhalsBinary, {}),
            (ZNarwhalsDecimal, {}),
            (ZNarwhalsFloat32, {}),
            (ZNarwhalsFloat64, {}),
            (ZNarwhalsInt8, {}),
            (ZNarwhalsInt16, {}),
            (ZNarwhalsInt32, {}),
            (ZNarwhalsInt64, {}),
            (ZNarwhalsList, {}),
            (ZNarwhalsArray, {}),
            (ZNarwhalsStruct, {}),
            (ZNarwhalsObject, {}),
            (ZNarwhalsUnknown, {}),
            (ZNarwhalsCategoricalCodes, {}),
            (ZNarwhalsCategoricalCategories, {"inner_dtype": "string"}),
            (ZNarwhalsEnum, {"categories": ("A", "B")}),
        ],
    )
    def test_v2_to_json_raises(self, dtype_cls, kwargs):
        """Test that to_json raises for Zarr v2."""
        dtype = dtype_cls(**kwargs)
        with pytest.raises(ValueError, match=r"only supports Zarr v3"):
            dtype.to_json(zarr_format=2)

    @pytest.mark.parametrize(
        "dtype_cls",
        [
            ZNarwhalsDate,
            ZNarwhalsTime,
            ZNarwhalsDatetime,
            ZNarwhalsDuration,
            ZNarwhalsBinary,
            ZNarwhalsDecimal,
            ZNarwhalsFloat32,
            ZNarwhalsFloat64,
            ZNarwhalsInt8,
            ZNarwhalsInt16,
            ZNarwhalsInt32,
            ZNarwhalsInt64,
            ZNarwhalsList,
            ZNarwhalsArray,
            ZNarwhalsStruct,
            ZNarwhalsObject,
            ZNarwhalsUnknown,
            ZNarwhalsCategoricalCodes,
            ZNarwhalsCategoricalCategories,
            ZNarwhalsEnum,
        ],
    )
    def test_v2_check_json_returns_false(self, dtype_cls):
        """Test that _check_json_v2 returns False."""
        assert dtype_cls._check_json_v2({}) is False
        assert dtype_cls._check_json_v2("test") is False

    @pytest.mark.parametrize(
        "dtype_cls",
        [
            ZNarwhalsDate,
            ZNarwhalsTime,
            ZNarwhalsDatetime,
            ZNarwhalsDuration,
            ZNarwhalsBinary,
            ZNarwhalsDecimal,
            ZNarwhalsFloat32,
            ZNarwhalsFloat64,
            ZNarwhalsInt8,
            ZNarwhalsInt16,
            ZNarwhalsInt32,
            ZNarwhalsInt64,
            ZNarwhalsList,
            ZNarwhalsArray,
            ZNarwhalsStruct,
            ZNarwhalsObject,
            ZNarwhalsUnknown,
            ZNarwhalsCategoricalCodes,
            ZNarwhalsCategoricalCategories,
            ZNarwhalsEnum,
        ],
    )
    def test_v2_from_json_raises(self, dtype_cls):
        """Test that _from_json_v2 raises DataTypeValidationError."""
        with pytest.raises(DataTypeValidationError, match=r"only supports Zarr v3|not v2"):
            dtype_cls._from_json_v2({})


class TestZDTypeNativeInference:
    """Test that ZDTypes prevent automatic inference from numpy dtypes."""

    @pytest.mark.parametrize(
        "dtype_cls",
        [
            ZNarwhalsDate,
            ZNarwhalsTime,
            ZNarwhalsDatetime,
            ZNarwhalsDuration,
            ZNarwhalsBinary,
            ZNarwhalsDecimal,
            ZNarwhalsFloat32,
            ZNarwhalsFloat64,
            ZNarwhalsInt8,
            ZNarwhalsInt16,
            ZNarwhalsInt32,
            ZNarwhalsInt64,
            ZNarwhalsList,
            ZNarwhalsArray,
            ZNarwhalsStruct,
            ZNarwhalsObject,
            ZNarwhalsUnknown,
            ZNarwhalsCategoricalCodes,
            ZNarwhalsCategoricalCategories,
            ZNarwhalsEnum,
        ],
    )
    def test_from_native_dtype_raises(self, dtype_cls):
        """Test that from_native_dtype raises DataTypeValidationError."""
        with pytest.raises(DataTypeValidationError, match="cannot be inferred"):
            dtype_cls.from_native_dtype(np.dtype("int32"))
