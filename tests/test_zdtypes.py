"""Comprehensive tests for ZDType implementations.

Tests all custom Zarr v3 data types in the zarrwhals.zdtypes module.
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
    ZNarwhalsBoolean,
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
    ZNarwhalsString,
    ZNarwhalsStruct,
    ZNarwhalsTime,
    ZNarwhalsUInt8,
    ZNarwhalsUInt16,
    ZNarwhalsUInt32,
    ZNarwhalsUInt64,
    ZNarwhalsUnknown,
)

SIMPLE_ZDTYPES = [
    pytest.param(ZNarwhalsFloat32, "narwhals.float32", np.dtype("float32"), np.float32(0.0), id="Float32"),
    pytest.param(ZNarwhalsFloat64, "narwhals.float64", np.dtype("float64"), np.float64(0.0), id="Float64"),
    pytest.param(ZNarwhalsInt8, "narwhals.int8", np.dtype("int8"), np.int8(0), id="Int8"),
    pytest.param(ZNarwhalsInt16, "narwhals.int16", np.dtype("int16"), np.int16(0), id="Int16"),
    pytest.param(ZNarwhalsInt32, "narwhals.int32", np.dtype("int32"), np.int32(0), id="Int32"),
    pytest.param(ZNarwhalsInt64, "narwhals.int64", np.dtype("int64"), np.int64(0), id="Int64"),
    pytest.param(ZNarwhalsUInt8, "narwhals.uint8", np.dtype("uint8"), np.uint8(0), id="UInt8"),
    pytest.param(ZNarwhalsUInt16, "narwhals.uint16", np.dtype("uint16"), np.uint16(0), id="UInt16"),
    pytest.param(ZNarwhalsUInt32, "narwhals.uint32", np.dtype("uint32"), np.uint32(0), id="UInt32"),
    pytest.param(ZNarwhalsUInt64, "narwhals.uint64", np.dtype("uint64"), np.uint64(0), id="UInt64"),
    pytest.param(ZNarwhalsBoolean, "narwhals.boolean", np.dtype("bool"), np.bool_(False), id="Boolean"),
    pytest.param(ZNarwhalsBinary, "narwhals.binary", np.dtype("O"), b"", id="Binary"),
    pytest.param(ZNarwhalsObject, "narwhals.object", np.dtype("O"), "null", id="Object"),
    pytest.param(ZNarwhalsUnknown, "narwhals.unknown", np.dtype("O"), "null", id="Unknown"),
    pytest.param(ZNarwhalsDate, "narwhals.date", np.dtype("datetime64[D]"), np.int32(0), id="Date"),
]

ALL_ZDTYPES_FOR_V2 = [
    pytest.param(ZNarwhalsDate, {}, id="ZNarwhalsDate"),
    pytest.param(ZNarwhalsTime, {}, id="ZNarwhalsTime"),
    pytest.param(ZNarwhalsDatetime, {}, id="ZNarwhalsDatetime"),
    pytest.param(ZNarwhalsDuration, {}, id="ZNarwhalsDuration"),
    pytest.param(ZNarwhalsBinary, {}, id="ZNarwhalsBinary"),
    pytest.param(ZNarwhalsBoolean, {}, id="ZNarwhalsBoolean"),
    pytest.param(ZNarwhalsDecimal, {}, id="ZNarwhalsDecimal"),
    pytest.param(ZNarwhalsFloat32, {}, id="ZNarwhalsFloat32"),
    pytest.param(ZNarwhalsFloat64, {}, id="ZNarwhalsFloat64"),
    pytest.param(ZNarwhalsInt8, {}, id="ZNarwhalsInt8"),
    pytest.param(ZNarwhalsInt16, {}, id="ZNarwhalsInt16"),
    pytest.param(ZNarwhalsInt32, {}, id="ZNarwhalsInt32"),
    pytest.param(ZNarwhalsInt64, {}, id="ZNarwhalsInt64"),
    pytest.param(ZNarwhalsUInt8, {}, id="ZNarwhalsUInt8"),
    pytest.param(ZNarwhalsUInt16, {}, id="ZNarwhalsUInt16"),
    pytest.param(ZNarwhalsUInt32, {}, id="ZNarwhalsUInt32"),
    pytest.param(ZNarwhalsUInt64, {}, id="ZNarwhalsUInt64"),
    pytest.param(ZNarwhalsList, {}, id="ZNarwhalsList"),
    pytest.param(ZNarwhalsArray, {}, id="ZNarwhalsArray"),
    pytest.param(ZNarwhalsString, {}, id="ZNarwhalsString"),
    pytest.param(ZNarwhalsStruct, {}, id="ZNarwhalsStruct"),
    pytest.param(ZNarwhalsObject, {}, id="ZNarwhalsObject"),
    pytest.param(ZNarwhalsUnknown, {}, id="ZNarwhalsUnknown"),
    pytest.param(ZNarwhalsCategoricalCodes, {}, id="ZNarwhalsCategoricalCodes"),
    pytest.param(ZNarwhalsCategoricalCategories, {"inner_dtype": "string"}, id="ZNarwhalsCategoricalCategories"),
    pytest.param(ZNarwhalsEnum, {"categories": ("A", "B")}, id="ZNarwhalsEnum"),
]

ALL_ZDTYPE_CLASSES = [
    ZNarwhalsDate,
    ZNarwhalsTime,
    ZNarwhalsDatetime,
    ZNarwhalsDuration,
    ZNarwhalsBinary,
    ZNarwhalsBoolean,
    ZNarwhalsDecimal,
    ZNarwhalsFloat32,
    ZNarwhalsFloat64,
    ZNarwhalsInt8,
    ZNarwhalsInt16,
    ZNarwhalsInt32,
    ZNarwhalsInt64,
    ZNarwhalsUInt8,
    ZNarwhalsUInt16,
    ZNarwhalsUInt32,
    ZNarwhalsUInt64,
    ZNarwhalsList,
    ZNarwhalsArray,
    ZNarwhalsString,
    ZNarwhalsStruct,
    ZNarwhalsObject,
    ZNarwhalsUnknown,
    ZNarwhalsCategoricalCodes,
    ZNarwhalsCategoricalCategories,
    ZNarwhalsEnum,
]


class TestSimpleZDTypes:
    """Parameterized tests for simple ZDTypes with no configuration."""

    @pytest.mark.parametrize(("dtype_cls", "expected_name", "native", "default"), SIMPLE_ZDTYPES)
    def test_json_v3_roundtrip(self, dtype_cls, expected_name, native, default):
        """Test JSON serialization produces correct name and roundtrips."""
        dtype = dtype_cls()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == expected_name
        assert result["configuration"] == {}

        restored = dtype_cls._from_json_v3(result)
        assert restored == dtype

    @pytest.mark.parametrize(("dtype_cls", "expected_name", "native", "default"), SIMPLE_ZDTYPES)
    def test_to_native_dtype(self, dtype_cls, expected_name, native, default):
        """Test conversion to native NumPy dtype."""
        dtype = dtype_cls()
        assert dtype.to_native_dtype() == native

    @pytest.mark.parametrize(("dtype_cls", "expected_name", "native", "default"), SIMPLE_ZDTYPES)
    def test_default_scalar(self, dtype_cls, expected_name, native, default):
        """Test default scalar value."""
        dtype = dtype_cls()
        result = dtype.default_scalar()
        assert result == default


class TestNumericCastScalar:
    """Test cast_scalar for numeric types."""

    @pytest.mark.parametrize(
        ("dtype_cls", "np_type"),
        [
            (ZNarwhalsFloat32, np.float32),
            (ZNarwhalsFloat64, np.float64),
            (ZNarwhalsInt8, np.int8),
            (ZNarwhalsInt16, np.int16),
            (ZNarwhalsInt32, np.int32),
            (ZNarwhalsInt64, np.int64),
            (ZNarwhalsUInt8, np.uint8),
            (ZNarwhalsUInt16, np.uint16),
            (ZNarwhalsUInt32, np.uint32),
            (ZNarwhalsUInt64, np.uint64),
        ],
    )
    def test_cast_scalar_int(self, dtype_cls, np_type):
        """Test casting integer values."""
        dtype = dtype_cls()
        assert dtype.cast_scalar(42) == np_type(42)

    @pytest.mark.parametrize(
        ("dtype_cls", "np_type"),
        [
            (ZNarwhalsFloat32, np.float32),
            (ZNarwhalsFloat64, np.float64),
        ],
    )
    def test_cast_scalar_float(self, dtype_cls, np_type):
        """Test casting float values."""
        dtype = dtype_cls()
        assert dtype.cast_scalar(1.5) == np_type(1.5)


class TestZNarwhalsTime:
    """Tests for ZNarwhalsTime dtype with time_unit configuration."""

    @pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
    def test_json_v3_time_units(self, time_unit):
        """Test JSON serialization preserves time_unit."""
        dtype = ZNarwhalsTime(time_unit=time_unit)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.time"
        assert result["configuration"]["time_unit"] == time_unit

        restored = ZNarwhalsTime._from_json_v3(result)
        assert restored.time_unit == time_unit

    def test_to_native_dtype(self):
        """Time stores as int64 regardless of unit."""
        dtype = ZNarwhalsTime(time_unit="us")
        assert dtype.to_native_dtype() == np.dtype("int64")

    def test_default_scalar(self):
        """Default is midnight (0)."""
        dtype = ZNarwhalsTime()
        assert dtype.default_scalar() == np.int64(0)


class TestZNarwhalsDatetime:
    """Tests for ZNarwhalsDatetime dtype with time_unit and time_zone."""

    @pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
    def test_json_v3_time_units(self, time_unit):
        """Test JSON serialization preserves time_unit."""
        dtype = ZNarwhalsDatetime(time_unit=time_unit)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["time_unit"] == time_unit

    @pytest.mark.parametrize("time_zone", [None, "UTC", "America/New_York", "Europe/London"])
    def test_json_v3_timezones(self, time_zone):
        """Test JSON serialization preserves time_zone."""
        dtype = ZNarwhalsDatetime(time_zone=time_zone)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["time_zone"] == time_zone

    def test_roundtrip_with_timezone(self):
        """Test full roundtrip with timezone."""
        dtype = ZNarwhalsDatetime(time_unit="ms", time_zone="America/New_York")
        json_data = dtype.to_json(zarr_format=3)
        restored = ZNarwhalsDatetime._from_json_v3(json_data)
        assert restored == dtype

    @pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
    def test_to_native_dtype(self, time_unit):
        """Test native dtype matches time_unit."""
        dtype = ZNarwhalsDatetime(time_unit=time_unit)
        assert dtype.to_native_dtype() == np.dtype(f"datetime64[{time_unit}]")

    def test_default_scalar(self):
        """Default is epoch (0)."""
        dtype = ZNarwhalsDatetime()
        assert dtype.default_scalar() == np.int64(0)


class TestZNarwhalsDuration:
    """Tests for ZNarwhalsDuration dtype with time_unit."""

    @pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
    def test_json_v3_time_units(self, time_unit):
        """Test JSON serialization preserves time_unit."""
        dtype = ZNarwhalsDuration(time_unit=time_unit)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.duration"
        assert result["configuration"]["time_unit"] == time_unit

        restored = ZNarwhalsDuration._from_json_v3(result)
        assert restored == dtype

    @pytest.mark.parametrize("time_unit", ["ns", "us", "ms", "s"])
    def test_to_native_dtype(self, time_unit):
        """Test native dtype matches time_unit."""
        dtype = ZNarwhalsDuration(time_unit=time_unit)
        assert dtype.to_native_dtype() == np.dtype(f"timedelta64[{time_unit}]")


class TestZNarwhalsDecimal:
    """Tests for ZNarwhalsDecimal dtype with precision and scale."""

    def test_json_v3_defaults(self):
        """Test JSON with default precision/scale."""
        dtype = ZNarwhalsDecimal()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.decimal"
        assert result["configuration"]["precision"] is None
        assert result["configuration"]["scale"] is None

    @pytest.mark.parametrize(
        ("precision", "scale"),
        [(10, 2), (18, 6), (38, 10)],
    )
    def test_json_v3_with_precision_scale(self, precision, scale):
        """Test JSON serialization with precision and scale."""
        dtype = ZNarwhalsDecimal(precision=precision, scale=scale)
        result = dtype.to_json(zarr_format=3)
        assert result["configuration"]["precision"] == precision
        assert result["configuration"]["scale"] == scale

        restored = ZNarwhalsDecimal._from_json_v3(result)
        assert restored == dtype


class TestZNarwhalsList:
    """Tests for ZNarwhalsList dtype with inner_dtype."""

    @pytest.mark.parametrize("inner_dtype", ["Int64", "String", "Float64", "Boolean"])
    def test_json_v3_inner_dtypes(self, inner_dtype):
        """Test JSON serialization with various inner dtypes."""
        dtype = ZNarwhalsList(inner_dtype=inner_dtype)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.list"
        assert result["configuration"]["inner_dtype"] == inner_dtype

        restored = ZNarwhalsList._from_json_v3(result)
        assert restored == dtype

    def test_to_native_dtype(self):
        """List stores as object dtype."""
        dtype = ZNarwhalsList(inner_dtype="Float64")
        assert dtype.to_native_dtype() == np.dtype("O")

    def test_default_scalar(self):
        """Default is empty list JSON."""
        dtype = ZNarwhalsList()
        assert dtype.default_scalar() == "[]"


class TestZNarwhalsArray:
    """Tests for ZNarwhalsArray dtype with inner_dtype and shape."""

    @pytest.mark.parametrize(
        ("inner_dtype", "shape"),
        [
            ("Float32", (10,)),
            ("Float64", (5,)),
            ("Int16", (3, 4)),
            ("Int32", (2, 3, 4)),
        ],
    )
    def test_json_v3_configurations(self, inner_dtype, shape):
        """Test JSON serialization with various configurations."""
        dtype = ZNarwhalsArray(inner_dtype=inner_dtype, shape=shape)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.array"
        assert result["configuration"]["inner_dtype"] == inner_dtype
        assert result["configuration"]["shape"] == list(shape)

        restored = ZNarwhalsArray._from_json_v3(result)
        assert restored.inner_dtype == dtype.inner_dtype
        assert restored.shape == dtype.shape

    def test_to_native_dtype(self):
        """Array native dtype based on inner_dtype."""
        dtype = ZNarwhalsArray(inner_dtype="Float64", shape=(5,))
        assert dtype.to_native_dtype() == np.dtype("float64")

    def test_default_scalar(self):
        """Default is zeros array of correct shape."""
        dtype = ZNarwhalsArray(inner_dtype="Float32", shape=(3,))
        result = dtype.default_scalar()
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))


class TestZNarwhalsStruct:
    """Tests for ZNarwhalsStruct dtype with fields."""

    @pytest.mark.parametrize(
        "fields",
        [
            (("name", "String"), ("age", "Int32")),
            (("x", "Float64"), ("y", "Float64")),
            (("a", "Int64"),),
        ],
    )
    def test_json_v3_fields(self, fields):
        """Test JSON serialization with various field configurations."""
        dtype = ZNarwhalsStruct(fields=fields)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.struct"
        expected_fields = [{"name": f[0], "dtype": f[1]} for f in fields]
        assert result["configuration"]["fields"] == expected_fields

        restored = ZNarwhalsStruct._from_json_v3(result)
        assert restored.fields == dtype.fields

    def test_to_native_dtype(self):
        """Struct stores as object dtype."""
        dtype = ZNarwhalsStruct(fields=(("a", "Int64"),))
        assert dtype.to_native_dtype() == np.dtype("O")

    def test_default_scalar(self):
        """Default is empty object JSON."""
        dtype = ZNarwhalsStruct()
        assert dtype.default_scalar() == "{}"


class TestZNarwhalsCategoricalCodes:
    """Tests for ZNarwhalsCategoricalCodes dtype."""

    @pytest.mark.parametrize("ordered", [True, False])
    def test_json_v3_ordered(self, ordered):
        """Test JSON serialization with ordered flag."""
        dtype = ZNarwhalsCategoricalCodes(ordered=ordered)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.categorical.codes"
        assert result["configuration"]["ordered"] is ordered

        restored = ZNarwhalsCategoricalCodes._from_json_v3(result)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Codes store as int32."""
        dtype = ZNarwhalsCategoricalCodes()
        assert dtype.to_native_dtype() == np.dtype("int32")

    def test_default_scalar(self):
        """Default is -1 (missing)."""
        dtype = ZNarwhalsCategoricalCodes()
        assert dtype.default_scalar() == np.int32(-1)


class TestZNarwhalsCategoricalCategories:
    """Tests for ZNarwhalsCategoricalCategories dtype."""

    @pytest.mark.parametrize(
        ("inner_dtype", "expected_native"),
        [
            ("string", np.dtypes.StringDType()),
            ("int64", np.dtype("int64")),
            ("float64", np.dtype("float64")),
        ],
    )
    def test_json_v3_inner_dtypes(self, inner_dtype, expected_native):
        """Test JSON serialization and native dtype for inner types."""
        dtype = ZNarwhalsCategoricalCategories(inner_dtype=inner_dtype)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.categorical.categories"
        assert result["configuration"]["inner_dtype"] == inner_dtype

        restored = ZNarwhalsCategoricalCategories._from_json_v3(result)
        assert restored == dtype
        assert dtype.to_native_dtype() == expected_native


class TestZNarwhalsEnum:
    """Tests for ZNarwhalsEnum dtype."""

    @pytest.mark.parametrize(
        ("categories", "ordered"),
        [
            (("A", "B", "C"), True),
            (("X", "Y"), False),
            (("small", "medium", "large"), True),
        ],
    )
    def test_json_v3_configurations(self, categories, ordered):
        """Test JSON serialization with various configurations."""
        dtype = ZNarwhalsEnum(categories=categories, ordered=ordered)
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.enum"
        assert result["configuration"]["categories"] == list(categories)
        assert result["configuration"]["ordered"] is ordered

        restored = ZNarwhalsEnum._from_json_v3(result)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Enum codes store as int32."""
        dtype = ZNarwhalsEnum(categories=("X", "Y"))
        assert dtype.to_native_dtype() == np.dtype("int32")

    def test_default_scalar(self):
        """Default is -1 (missing)."""
        dtype = ZNarwhalsEnum(categories=("A", "B"))
        assert dtype.default_scalar() == np.int32(-1)


class TestZDTypeV2Rejection:
    """Test that all ZDTypes properly reject Zarr v2 format."""

    @pytest.mark.parametrize(("dtype_cls", "kwargs"), ALL_ZDTYPES_FOR_V2)
    def test_v2_to_json_raises(self, dtype_cls, kwargs):
        """Test that to_json raises for Zarr v2."""
        dtype = dtype_cls(**kwargs)
        with pytest.raises(ValueError, match=r"only supports Zarr v3"):
            dtype.to_json(zarr_format=2)

    @pytest.mark.parametrize("dtype_cls", ALL_ZDTYPE_CLASSES)
    def test_v2_check_json_returns_false(self, dtype_cls):
        """Test that _check_json_v2 returns False for any input."""
        assert dtype_cls._check_json_v2({}) is False
        assert dtype_cls._check_json_v2("test") is False

    @pytest.mark.parametrize("dtype_cls", ALL_ZDTYPE_CLASSES)
    def test_v2_from_json_raises(self, dtype_cls):
        """Test that _from_json_v2 raises DataTypeValidationError."""
        with pytest.raises(DataTypeValidationError, match=r"only supports Zarr v3|not v2"):
            dtype_cls._from_json_v2({})


class TestZDTypeNativeInference:
    """Test that ZDTypes prevent automatic inference from numpy dtypes."""

    @pytest.mark.parametrize("dtype_cls", ALL_ZDTYPE_CLASSES)
    def test_from_native_dtype_raises(self, dtype_cls):
        """Test that from_native_dtype raises DataTypeValidationError."""
        with pytest.raises(DataTypeValidationError, match="cannot be inferred"):
            dtype_cls.from_native_dtype(np.dtype("int32"))


class TestZNarwhalsBoolean:
    """Tests for ZNarwhalsBoolean dtype."""

    def test_json_v3_roundtrip(self):
        """Test JSON serialization produces correct name and roundtrips."""
        dtype = ZNarwhalsBoolean()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.boolean"
        assert result["configuration"] == {}

        restored = ZNarwhalsBoolean._from_json_v3(result)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to native NumPy dtype."""
        dtype = ZNarwhalsBoolean()
        assert dtype.to_native_dtype() == np.dtype("bool")

    def test_cast_scalar(self):
        """Test casting boolean values."""
        dtype = ZNarwhalsBoolean()
        assert dtype.cast_scalar(True) == np.bool_(True)
        assert dtype.cast_scalar(False) == np.bool_(False)

    def test_default_scalar(self):
        """Default is False."""
        dtype = ZNarwhalsBoolean()
        assert dtype.default_scalar() == np.bool_(False)

    def test_cast_invalid_raises(self):
        """Test that invalid types raise TypeError."""
        dtype = ZNarwhalsBoolean()
        with pytest.raises(TypeError):
            dtype.cast_scalar("not a bool")


class TestZNarwhalsString:
    """Tests for ZNarwhalsString dtype."""

    def test_json_v3_roundtrip(self):
        """Test JSON serialization produces correct name and roundtrips."""
        dtype = ZNarwhalsString()
        result = dtype.to_json(zarr_format=3)
        assert result["name"] == "narwhals.string"
        assert result["configuration"] == {}

        restored = ZNarwhalsString._from_json_v3(result)
        assert restored == dtype

    def test_to_native_dtype(self):
        """Test conversion to native NumPy dtype."""
        from numpy.dtypes import StringDType

        dtype = ZNarwhalsString()
        assert dtype.to_native_dtype() == np.dtype(StringDType())

    def test_cast_scalar(self):
        """Test casting string values."""
        dtype = ZNarwhalsString()
        assert dtype.cast_scalar("hello") == "hello"
        assert dtype.cast_scalar("") == ""

    def test_default_scalar(self):
        """Default is empty string."""
        dtype = ZNarwhalsString()
        assert dtype.default_scalar() == ""

    def test_cast_invalid_raises(self):
        """Test that invalid types raise TypeError."""
        dtype = ZNarwhalsString()
        with pytest.raises(TypeError):
            dtype.cast_scalar(123)


class TestZDTypeErrorHandling:
    """Test error handling for invalid inputs."""

    def test_date_invalid_json(self):
        """Test that invalid JSON raises error."""
        with pytest.raises(DataTypeValidationError):
            ZNarwhalsDate._from_json_v3({"name": "wrong"})

    def test_date_cast_scalar(self):
        """Test casting integer to date scalar."""
        dtype = ZNarwhalsDate()
        assert dtype.cast_scalar(100) == np.int32(100)
