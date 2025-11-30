"""Custom Zarr v3 Data Types for Narwhals dtypes.

This module provides custom ZDType implementations for Narwhals data types,
enabling proper Zarr v3 metadata encoding and registration following ZEP0009.

The custom dtypes use the "narwhals.*" namespace for Zarr registry integration.
"""

from __future__ import annotations

from zarr.core.dtype import data_type_registry

from .array import ZNarwhalsArray
from .binary import ZNarwhalsBinary
from .categorical import (
    ZNarwhalsCategoricalCategories,
    ZNarwhalsCategoricalCodes,
    get_categorical_codes,
)
from .converters import zdtype_to_narwhals
from .date import ZNarwhalsDate
from .datetime import ZNarwhalsDatetime
from .decimal import ZNarwhalsDecimal
from .duration import ZNarwhalsDuration
from .enum import ZNarwhalsEnum
from .float32 import ZNarwhalsFloat32
from .float64 import ZNarwhalsFloat64
from .int8 import ZNarwhalsInt8
from .int16 import ZNarwhalsInt16
from .int32 import ZNarwhalsInt32
from .int64 import ZNarwhalsInt64
from .list import ZNarwhalsList
from .object import ZNarwhalsObject
from .struct import ZNarwhalsStruct
from .time import ZNarwhalsTime
from .unknown import ZNarwhalsUnknown

data_type_registry.register(ZNarwhalsArray._zarr_v3_name, ZNarwhalsArray)
data_type_registry.register(ZNarwhalsBinary._zarr_v3_name, ZNarwhalsBinary)
data_type_registry.register(ZNarwhalsCategoricalCodes._zarr_v3_name, ZNarwhalsCategoricalCodes)
data_type_registry.register(ZNarwhalsCategoricalCategories._zarr_v3_name, ZNarwhalsCategoricalCategories)
data_type_registry.register(ZNarwhalsDate._zarr_v3_name, ZNarwhalsDate)
data_type_registry.register(ZNarwhalsDatetime._zarr_v3_name, ZNarwhalsDatetime)
data_type_registry.register(ZNarwhalsDecimal._zarr_v3_name, ZNarwhalsDecimal)
data_type_registry.register(ZNarwhalsDuration._zarr_v3_name, ZNarwhalsDuration)
data_type_registry.register(ZNarwhalsEnum._zarr_v3_name, ZNarwhalsEnum)
data_type_registry.register(ZNarwhalsFloat32._zarr_v3_name, ZNarwhalsFloat32)
data_type_registry.register(ZNarwhalsFloat64._zarr_v3_name, ZNarwhalsFloat64)
data_type_registry.register(ZNarwhalsInt8._zarr_v3_name, ZNarwhalsInt8)
data_type_registry.register(ZNarwhalsInt16._zarr_v3_name, ZNarwhalsInt16)
data_type_registry.register(ZNarwhalsInt32._zarr_v3_name, ZNarwhalsInt32)
data_type_registry.register(ZNarwhalsInt64._zarr_v3_name, ZNarwhalsInt64)
data_type_registry.register(ZNarwhalsList._zarr_v3_name, ZNarwhalsList)
data_type_registry.register(ZNarwhalsObject._zarr_v3_name, ZNarwhalsObject)
data_type_registry.register(ZNarwhalsStruct._zarr_v3_name, ZNarwhalsStruct)
data_type_registry.register(ZNarwhalsTime._zarr_v3_name, ZNarwhalsTime)
data_type_registry.register(ZNarwhalsUnknown._zarr_v3_name, ZNarwhalsUnknown)

__all__ = [
    "ZNarwhalsArray",
    "ZNarwhalsBinary",
    "ZNarwhalsCategoricalCategories",
    "ZNarwhalsCategoricalCodes",
    "ZNarwhalsDate",
    "ZNarwhalsDatetime",
    "ZNarwhalsDecimal",
    "ZNarwhalsDuration",
    "ZNarwhalsEnum",
    "ZNarwhalsFloat32",
    "ZNarwhalsFloat64",
    "ZNarwhalsInt8",
    "ZNarwhalsInt16",
    "ZNarwhalsInt32",
    "ZNarwhalsInt64",
    "ZNarwhalsList",
    "ZNarwhalsObject",
    "ZNarwhalsStruct",
    "ZNarwhalsTime",
    "ZNarwhalsUnknown",
    "get_categorical_codes",
    "zdtype_to_narwhals",
]
