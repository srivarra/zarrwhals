"""Narwhals Series to Zarr array writers.

Handles pre-processing of Narwhals Series data before Zarr's codec pipeline.
Works directly with Narwhals Series using series.to_numpy() for data extraction.
"""

from __future__ import annotations

from typing import Literal

import narwhals as nw
import numpy as np
from zarr import Array, Group
from zarr.core.array import CompressorsLike
from zarr.core.dtype import VariableLengthUTF8

from .specs import CategoricalGroupAttributes, ColumnArrayAttributes


def _normalize_shape(size: int | Literal["auto"] | None, ndim: int) -> tuple[int, ...] | Literal["auto"] | None:
    """Normalize size to tuple shape for Zarr arrays.

    Parameters
    ----------
    size
        Size parameter (int, "auto", tuple, or None).
    ndim
        Number of dimensions.

    Returns
    -------
    tuple of int, "auto", or None
        Normalized shape, "auto", or None.
    """
    if size is None or size == "auto":
        return size
    if isinstance(size, tuple):
        return size
    return (size,) * ndim


def _set_column_attrs(arr: Array, encoding_type: str, dtype: nw.DType) -> None:
    """Set column array attributes.

    Parameters
    ----------
    arr
        Zarr array.
    encoding_type
        Encoding type (e.g., "numeric-array").
    dtype
        Narwhals dtype.
    """
    attrs = ColumnArrayAttributes(encoding_type=encoding_type, narwhals_dtype=str(dtype))
    arr.attrs.update(attrs.model_dump(by_alias=True, exclude_none=True))


def _create_zarr_array(
    group: Group,
    name: str,
    data: np.ndarray,
    zarr_dtype,
    encoding_type: str,
    narwhals_dtype: nw.DType,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
    extra_attrs: dict | None = None,
) -> Array:
    """Create Zarr array with common boilerplate handling.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Prepared data to write.
    zarr_dtype
        Zarr dtype (can be custom ZDType).
    encoding_type
        Encoding type for column attributes.
    narwhals_dtype
        Original Narwhals dtype.
    chunks
        Chunk size.
    shards
        Shard size.
    compressors
        Compressor codec(s).
    extra_attrs
        Additional attributes to set on array.

    Returns
    -------
    Array
        Created and populated Zarr array.
    """
    chunk_shape = _normalize_shape(chunks, data.ndim)
    shard_shape = _normalize_shape(shards, data.ndim)

    arr = group.create_array(
        name,
        shape=data.shape,
        dtype=zarr_dtype,
        chunks=chunk_shape,
        shards=shard_shape,
        compressors=compressors,
    )
    _set_column_attrs(arr, encoding_type, narwhals_dtype)
    if extra_attrs:
        arr.attrs.update(extra_attrs)
    arr[...] = data
    return arr


def _set_categorical_attrs(group: Group, encoding_type: str, ordered: bool) -> None:
    """Set categorical/enum group attributes.

    Parameters
    ----------
    group
        Zarr group.
    encoding_type
        "categorical" or "enum".
    ordered
        Whether ordered.
    """
    attrs = CategoricalGroupAttributes(encoding_type=encoding_type, ordered=ordered)  # type: ignore[arg-type]
    group.attrs.update(attrs.model_dump(by_alias=True))


def encode_series_narwhals(
    group: Group,
    name: str,
    series: nw.Series,
    *,
    chunks: int | Literal["auto"] | None = "auto",
    shards: int | None = None,
    compressors: CompressorsLike = "auto",
) -> Array | Group:
    """Encode Narwhals Series to Zarr using backend-agnostic approach.

    Parameters
    ----------
    group
        Zarr group to write into
    name
        Array/group name
    series
        Narwhals Series to encode
    chunks
        Chunk size for 1D arrays
    shards
        Shard size for 1D arrays
    compressors
        Compressor codec(s)

    Returns
    -------
    Array | Group
        Created Zarr array or group
    """
    dtype = series.dtype

    data = series.to_numpy()

    match dtype:
        case nw.Boolean:
            return _encode_numeric(group, name, data, dtype, chunks, shards, compressors)
        case nw.Categorical:
            return _encode_categorical(group, name, series, dtype, chunks, shards, compressors)
        case nw.Enum():
            return _encode_enum(group, name, series, dtype, chunks, shards, compressors)
        case nw.String():
            return _encode_string(group, name, data, dtype, chunks, shards, compressors)
        case nw.Datetime():
            return _encode_datetime(group, name, data, dtype, chunks, shards, compressors)
        case nw.Duration():
            return _encode_duration(group, name, data, dtype, chunks, shards, compressors)
        case nw.Date():
            return _encode_date(group, name, data, dtype, chunks, shards, compressors)
        case nw.Time():
            return _encode_time(group, name, data, dtype, chunks, shards, compressors)
        case nw.Binary():
            return _encode_binary(group, name, data, dtype, chunks, shards, compressors)
        case nw.Decimal():
            return _encode_decimal(group, name, data, dtype, chunks, shards, compressors)
        case nw.List():
            return _encode_list(group, name, series, dtype, chunks, shards, compressors)
        case nw.Struct():
            return _encode_struct(group, name, series, dtype, chunks, shards, compressors)
        case nw.Array():
            return _encode_array(group, name, series, dtype, chunks, shards, compressors)
        case nw.Object:
            return _encode_object(group, name, data, dtype, chunks, shards, compressors)
        case nw.Unknown:
            # For unknown types, try to encode as object array
            return _encode_object(group, name, data, dtype, chunks, shards, compressors)
        case _ if dtype.is_numeric():
            # Numeric types (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64)
            return _encode_numeric(group, name, data, dtype, chunks, shards, compressors)
        case _:
            msg = f"Unsupported Narwhals dtype: {dtype}"
            raise TypeError(msg)


def _encode_numeric(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.DType,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode numeric array to Zarr.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Numeric data (int, float, or bool).
    dtype
        Narwhals dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    return _create_zarr_array(
        group,
        name,
        data,
        zarr_dtype=data.dtype,
        encoding_type="numeric-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_string(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.DType,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode string array using variable-length UTF-8.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        String data.
    dtype
        Narwhals String dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.

    Notes
    -----
    Uses NumPy StringDType for efficient variable-length storage.
    """
    from numpy.dtypes import StringDType

    # Convert object dtype to StringDType for memory efficiency
    if data.dtype.kind == "O" or data.dtype.kind == "U":
        data = data.astype(StringDType())

    return _create_zarr_array(
        group,
        name,
        data,
        zarr_dtype=VariableLengthUTF8(),
        encoding_type="string-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_datetime(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Datetime,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode datetime array as int64 with timezone preservation.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Datetime64 data.
    dtype
        Narwhals Datetime dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.

    Notes
    -----
    Stores as int64 with time unit and optional timezone in metadata.
    """
    from .zdtypes import ZNarwhalsDatetime

    unit = np.datetime_data(data.dtype)[0]
    int_data = data.view("int64")
    time_zone = dtype.time_zone if hasattr(dtype, "time_zone") else None

    return _create_zarr_array(
        group,
        name,
        int_data,
        zarr_dtype=ZNarwhalsDatetime(time_unit=unit, time_zone=time_zone),
        encoding_type="datetime-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_duration(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Duration,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode duration/timedelta array as int64.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Timedelta64 data.
    dtype
        Narwhals Duration dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    from .zdtypes import ZNarwhalsDuration

    int_data = data.view("int64") if data.dtype.kind == "m" else data.astype("int64")
    unit = (
        np.datetime_data(data.dtype)[0]
        if data.dtype.kind == "m"
        else (dtype.time_unit if hasattr(dtype, "time_unit") else "ns")
    )

    return _create_zarr_array(
        group,
        name,
        int_data,
        zarr_dtype=ZNarwhalsDuration(time_unit=unit),
        encoding_type="duration-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_date(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Date,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode date array as int32 (days since epoch).

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Date data (datetime64[D]).
    dtype
        Narwhals Date dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    from .zdtypes import ZNarwhalsDate

    # Convert datetime64[D] to int32 view (days since epoch)
    if data.dtype.kind == "M":
        if np.datetime_data(data.dtype)[0] != "D":
            data = data.astype("datetime64[D]")
        int_data = data.view("int32")
    else:
        int_data = data.astype("int32")

    return _create_zarr_array(
        group,
        name,
        int_data,
        zarr_dtype=ZNarwhalsDate(),
        encoding_type="date-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_enum(
    group: Group,
    name: str,
    series: nw.Series,
    dtype: nw.Enum,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Group:
    """Encode Enum (fixed categories) as group with codes and categories.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Enum subgroup name.
    series
        Narwhals Series with Enum dtype.
    dtype
        Enum dtype with fixed categories.
    chunks
        Chunk size for codes array.
    shards
        Shard size for codes array.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Group
        Created enum group.

    Notes
    -----
    Categories from dtype (fixed), not inferred from data.
    """
    from .zdtypes import get_categorical_codes

    # Extract codes only - for Enum, categories come from dtype (not data)
    codes, _, _ = get_categorical_codes(series)

    # Create enum group (similar to AnnData categorical but with "enum" type)
    enum_group = group.create_group(name)
    _set_categorical_attrs(enum_group, "enum", ordered=True)  # Enums are inherently ordered

    # Detect category dtype from Enum categories
    from .zdtypes import ZNarwhalsCategoricalCategories, ZNarwhalsCategoricalCodes

    categories_array = np.array(dtype.categories)
    if np.issubdtype(categories_array.dtype, np.integer):
        inner_dtype = "int64"
        categories_array = categories_array.astype(np.int64)
    elif np.issubdtype(categories_array.dtype, np.floating):
        inner_dtype = "float64"
        categories_array = categories_array.astype(np.float64)
    else:
        from numpy.dtypes import StringDType

        inner_dtype = "string"
        categories_array = categories_array.astype(StringDType())

    # Create custom dtypes
    codes_dtype = ZNarwhalsCategoricalCodes(ordered=False)  # Enums are not ordered by default
    cats_dtype = ZNarwhalsCategoricalCategories(inner_dtype=inner_dtype)

    # Store codes array with custom dtype
    codes_array = enum_group.create_array(
        "codes",
        shape=(len(codes),),
        dtype=codes_dtype,  # Type-safe custom dtype!
        chunks=_normalize_shape(chunks, 1),
        shards=_normalize_shape(shards, 1),
        compressors=compressors,
    )
    codes_array[:] = codes  # Write data after creation

    # Store categories array with custom dtype (preserves int/float/string!)
    # For string dtype, use direct data passing (StringDType has special handling)
    if inner_dtype == "string":
        enum_group.create_array("categories", data=categories_array)
    else:
        # For int64/float64, use custom dtype
        cats_array = enum_group.create_array("categories", shape=(len(categories_array),), dtype=cats_dtype)
        cats_array[:] = categories_array  # Write data after creation

    return enum_group


def _encode_categorical(
    group: Group,
    name: str,
    series: nw.Series,
    _dtype: nw.Categorical,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Group:
    """Encode Categorical (dynamic categories) as group with codes and categories.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Categorical subgroup name.
    series
        Narwhals Series with Categorical dtype.
    _dtype
        Categorical dtype (unused, kept for API consistency).
    chunks
        Chunk size for codes array.
    shards
        Shard size for codes array.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Group
        Created categorical group.

    Notes
    -----
    Categories extracted from data, not predefined in dtype.
    """
    from .zdtypes import ZNarwhalsCategoricalCategories, ZNarwhalsCategoricalCodes, get_categorical_codes

    # Extract codes and categories from data
    codes, categories, ordered = get_categorical_codes(series)

    # Detect category dtype (int64, float64, or string)
    categories_array = np.array(categories)
    if np.issubdtype(categories_array.dtype, np.integer):
        inner_dtype = "int64"
        categories_array = categories_array.astype(np.int64)
    elif np.issubdtype(categories_array.dtype, np.floating):
        inner_dtype = "float64"
        categories_array = categories_array.astype(np.float64)
    else:
        from numpy.dtypes import StringDType

        inner_dtype = "string"
        categories_array = categories_array.astype(StringDType())

    # Create custom dtypes
    codes_dtype = ZNarwhalsCategoricalCodes(ordered=ordered)
    cats_dtype = ZNarwhalsCategoricalCategories(inner_dtype=inner_dtype)

    # Create categorical group
    cat_group = group.create_group(name)
    _set_categorical_attrs(cat_group, "categorical", ordered=ordered)

    # Store codes array with custom dtype
    codes_array = cat_group.create_array(
        "codes",
        shape=(len(codes),),
        dtype=codes_dtype,  # Type-safe custom dtype!
        chunks=_normalize_shape(chunks, 1),
        shards=_normalize_shape(shards, 1),
        compressors=compressors,
    )
    codes_array[:] = codes  # Write data after creation

    # Store categories array with custom dtype (preserves int/float/string!)
    # For string dtype, use direct data passing (StringDType has special handling)
    if inner_dtype == "string":
        cat_group.create_array("categories", data=categories_array)
    else:
        # For int64/float64, use custom dtype
        cats_array = cat_group.create_array("categories", shape=(len(categories_array),), dtype=cats_dtype)
        cats_array[:] = categories_array  # Write data after creation

    return cat_group


def _encode_time(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Time,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode time-of-day array as int64 (nanoseconds since midnight).

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Time data (int64 nanoseconds).
    dtype
        Narwhals Time dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    from .zdtypes import ZNarwhalsTime

    int_data = data.astype("int64") if data.dtype != np.int64 else data

    return _create_zarr_array(
        group,
        name,
        int_data,
        zarr_dtype=ZNarwhalsTime(time_unit="ns"),
        encoding_type="time-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_binary(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Binary,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode binary data as base64 strings.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Binary data (object array of bytes).
    dtype
        Narwhals Binary dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    import base64

    from .zdtypes import ZNarwhalsBinary

    # Convert bytes to base64 strings for storage
    b64_data = np.array(
        [base64.b64encode(x).decode("ascii") if isinstance(x, bytes) else "" for x in data.flatten()],
        dtype=object,
    ).reshape(data.shape)

    return _create_zarr_array(
        group,
        name,
        b64_data,
        zarr_dtype=ZNarwhalsBinary(),
        encoding_type="binary-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_decimal(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.Decimal,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode decimal array as strings (preserves precision).

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Decimal data.
    dtype
        Narwhals Decimal dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    from .zdtypes import ZNarwhalsDecimal

    # Convert to string representation
    str_data = np.array([str(x) for x in data.flatten()], dtype=object).reshape(data.shape)
    precision = getattr(dtype, "precision", None)
    scale = getattr(dtype, "scale", None)

    return _create_zarr_array(
        group,
        name,
        str_data,
        zarr_dtype=ZNarwhalsDecimal(precision=precision, scale=scale),
        encoding_type="decimal-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )


def _encode_list(
    group: Group,
    name: str,
    series: nw.Series,
    dtype: nw.List,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode list array as JSON strings.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    series
        Narwhals Series with List dtype.
    dtype
        Narwhals List dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    import json

    from .zdtypes import ZNarwhalsList

    data = series.to_numpy()
    json_data = np.array([json.dumps(x.tolist() if hasattr(x, "tolist") else list(x)) for x in data], dtype=object)
    inner_dtype_str = str(dtype.inner)

    return _create_zarr_array(
        group,
        name,
        json_data,
        zarr_dtype=ZNarwhalsList(inner_dtype=inner_dtype_str),
        encoding_type="list-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        extra_attrs={"list_inner_dtype": inner_dtype_str},
    )


def _encode_struct(
    group: Group,
    name: str,
    series: nw.Series,
    dtype: nw.Struct,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode struct array as JSON strings.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    series
        Narwhals Series with Struct dtype.
    dtype
        Narwhals Struct dtype.
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    import json

    from .zdtypes import ZNarwhalsStruct

    data = series.to_numpy()
    json_data = np.array([json.dumps(x) if isinstance(x, dict) else json.dumps({}) for x in data], dtype=object)
    fields_tuple = tuple((f.name, str(f.dtype)) for f in dtype.fields)

    return _create_zarr_array(
        group,
        name,
        json_data,
        zarr_dtype=ZNarwhalsStruct(fields=fields_tuple),
        encoding_type="struct-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        extra_attrs={"struct_fields": json.dumps([{"name": f.name, "dtype": str(f.dtype)} for f in dtype.fields])},
    )


def _encode_array(
    group: Group,
    name: str,
    series: nw.Series,
    dtype: nw.Array,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode fixed-length array as multidimensional Zarr array.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    series
        Narwhals Series with Array dtype.
    dtype
        Narwhals Array dtype.
    chunks
        Chunk size for first dimension.
    shards
        Shard size for first dimension.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created multidimensional array.

    Notes
    -----
    Useful for embeddings, vectors, and small matrices per row.
    """
    from .zdtypes import ZNarwhalsArray

    # Extract as numpy array (should be 2D)
    data = series.to_numpy()
    if data.ndim == 1:
        if hasattr(data[0], "shape"):
            data = np.stack([np.asarray(x) for x in data])
        else:
            data = np.array([np.asarray(x).flatten() for x in data])
    elif data.ndim == 2:
        data = np.asarray([np.asarray(x) for x in data])

    inner_dtype_str = str(dtype.inner)
    array_shape = tuple(dtype.shape) if hasattr(dtype.shape, "__iter__") else (dtype.shape,)

    return _create_zarr_array(
        group,
        name,
        data,
        zarr_dtype=ZNarwhalsArray(inner_dtype=inner_dtype_str, shape=array_shape),
        encoding_type="array-array",
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        extra_attrs={"array_shape": str(array_shape), "array_inner_dtype": inner_dtype_str},
    )


def _encode_object(
    group: Group,
    name: str,
    data: np.ndarray,
    dtype: nw.DType,
    chunks: int | None,
    shards: int | None,
    compressors: CompressorsLike,
) -> Array:
    """Encode object array as JSON strings.

    Parameters
    ----------
    group
        Parent Zarr group.
    name
        Array name.
    data
        Object data.
    dtype
        Narwhals dtype (Object or Unknown).
    chunks
        Chunk size in rows.
    shards
        Shard size in rows.
    compressors
        Compressor codec(s).

    Returns
    -------
    zarr.Array
        Created array.
    """
    import json

    from .zdtypes import ZNarwhalsObject, ZNarwhalsUnknown

    json_data = np.array(
        [json.dumps(x, default=str) if not isinstance(x, str) else x for x in data.flatten()],
        dtype=object,
    ).reshape(data.shape)

    if dtype == nw.Unknown:
        zarr_dtype = ZNarwhalsUnknown()
        encoding_type = "unknown-array"
    else:
        zarr_dtype = ZNarwhalsObject()
        encoding_type = "object-array"

    return _create_zarr_array(
        group,
        name,
        json_data,
        zarr_dtype=zarr_dtype,
        encoding_type=encoding_type,
        narwhals_dtype=dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
    )
