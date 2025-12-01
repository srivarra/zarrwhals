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

    Uses the ZDType's nw_dtype property to get the corresponding Narwhals dtype.
    This is the primary decoding path for stores using custom ZDTypes.

    Parameters
    ----------
    zdtype : ZDType
        Zarr v3 ZDType instance with type configuration.

    Returns
    -------
    narwhals.DType or None
        Corresponding Narwhals dtype object, or None if the ZDType is not a
        custom Narwhals dtype (e.g., standard Zarr types).

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
    All ZNarwhals* types have an `nw_dtype` property that returns the corresponding
    Narwhals dtype.
    """
    # Duck typing: if the ZDType has an nw_dtype property, use it
    if hasattr(zdtype, "nw_dtype"):
        return zdtype.nw_dtype

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
