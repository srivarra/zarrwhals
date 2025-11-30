"""Data type conversion utilities for ZDType to Narwhals dtype.

This module provides utilities for converting between ZDType metadata and Narwhals dtype
objects. Used internally by ZarrFrame for proper type reconstruction from Zarr v3 stores.
"""

from __future__ import annotations

import narwhals as nw
from zarr.core.dtype import ZDType

__all__ = [
    "zdtype_to_narwhals",
]


def zdtype_to_narwhals(zdtype: ZDType) -> nw.DType | None:
    """Convert ZDType metadata to Narwhals dtype.

    Uses the ZDType's configuration to reconstruct the corresponding Narwhals dtype.
    This is the primary decoding path for stores using custom ZDTypes.

    Parameters
    ----------
    zdtype : ZDType
        Zarr v3 ZDType instance with type configuration.

    Returns
    -------
    narwhals.DType or None
        Corresponding Narwhals dtype object, or None if the ZDType is not a
        custom Narwhals dtype (e.g., standard Zarr numeric types).

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsDatetime, ZNarwhalsDuration
    >>> from zarrwhals.zdtypes.converters import zdtype_to_narwhals
    >>> zdtype_to_narwhals(ZNarwhalsDatetime(time_unit="ns", time_zone="UTC"))
    Datetime(time_unit='ns', time_zone='UTC')
    >>> zdtype_to_narwhals(ZNarwhalsDuration(time_unit="ms"))
    Duration(time_unit='ms')

    Notes
    -----
    - ZNarwhalsDatetime → nw.Datetime (with time_unit and time_zone)
    - ZNarwhalsDuration → nw.Duration (with time_unit)
    - ZNarwhalsDate → nw.Date
    - ZNarwhalsTime → nw.Time
    - ZNarwhalsBinary → nw.Binary
    - ZNarwhalsList → nw.List (with inner dtype)
    - ZNarwhalsStruct → nw.Struct (with fields)
    - ZNarwhalsArray → nw.Array (with inner dtype and shape)
    - ZNarwhalsDecimal → nw.Decimal (with precision and scale)
    - ZNarwhalsObject → nw.Object
    - ZNarwhalsUnknown → nw.Unknown
    - ZNarwhalsEnum → nw.Enum (categories from metadata)
    - ZNarwhalsCategoricalCodes → nw.Categorical
    - Other ZDTypes → None (no casting needed)

    """
    from . import (
        ZNarwhalsArray,
        ZNarwhalsBinary,
        ZNarwhalsCategoricalCodes,
        ZNarwhalsDate,
        ZNarwhalsDatetime,
        ZNarwhalsDecimal,
        ZNarwhalsDuration,
        ZNarwhalsEnum,
        ZNarwhalsList,
        ZNarwhalsObject,
        ZNarwhalsStruct,
        ZNarwhalsTime,
        ZNarwhalsUnknown,
    )

    # Temporal types
    if isinstance(zdtype, ZNarwhalsDatetime):
        kwargs = {"time_unit": zdtype.time_unit}
        if zdtype.time_zone:
            kwargs["time_zone"] = zdtype.time_zone
        return nw.Datetime(**kwargs)

    if isinstance(zdtype, ZNarwhalsDuration):
        return nw.Duration(time_unit=zdtype.time_unit)

    if isinstance(zdtype, ZNarwhalsDate):
        return nw.Date

    if isinstance(zdtype, ZNarwhalsTime):
        return nw.Time

    # Binary
    if isinstance(zdtype, ZNarwhalsBinary):
        return nw.Binary

    # Nested types
    if isinstance(zdtype, ZNarwhalsList):
        inner = _parse_inner_dtype(zdtype.inner_dtype)
        return nw.List(inner)

    if isinstance(zdtype, ZNarwhalsStruct):
        fields = {name: _parse_inner_dtype(dtype_str) for name, dtype_str in zdtype.fields}
        return nw.Struct(fields)

    if isinstance(zdtype, ZNarwhalsArray):
        inner = _parse_inner_dtype(zdtype.inner_dtype)
        return nw.Array(inner, zdtype.shape)

    # Decimal
    if isinstance(zdtype, ZNarwhalsDecimal):
        return nw.Decimal

    # Categorical types
    if isinstance(zdtype, ZNarwhalsEnum):
        return nw.Enum(zdtype.categories)

    if isinstance(zdtype, ZNarwhalsCategoricalCodes):
        return nw.Categorical

    # Fallback types
    if isinstance(zdtype, ZNarwhalsObject):
        return nw.Object

    if isinstance(zdtype, ZNarwhalsUnknown):
        return nw.Unknown

    # Unknown ZDType (e.g., standard Zarr numeric types) - return None to skip casting
    return None


def _parse_inner_dtype(dtype_str: str) -> nw.DType:
    """Parse simple Narwhals dtype from string representation.

    Used for inner dtypes in List, Struct, and Array types.

    Parameters
    ----------
    dtype_str : str
        String representation of Narwhals dtype (e.g., "Int64", "Float32", "String").

    Returns
    -------
    narwhals.DType
        Corresponding Narwhals dtype object.
    """
    simple_types = {
        "Int8": nw.Int8,
        "Int16": nw.Int16,
        "Int32": nw.Int32,
        "Int64": nw.Int64,
        "Int128": nw.Int128,
        "UInt8": nw.UInt8,
        "UInt16": nw.UInt16,
        "UInt32": nw.UInt32,
        "UInt64": nw.UInt64,
        "UInt128": nw.UInt128,
        "Float32": nw.Float32,
        "Float64": nw.Float64,
        "Boolean": nw.Boolean,
        "String": nw.String,
        "Categorical": nw.Categorical,
        "Date": nw.Date,
        "Time": nw.Time,
        "Binary": nw.Binary,
        "Decimal": nw.Decimal,
        "Object": nw.Object,
        "Unknown": nw.Unknown,
    }
    return simple_types.get(dtype_str, nw.String)
