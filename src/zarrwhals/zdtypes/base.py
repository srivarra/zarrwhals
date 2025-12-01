"""Base classes for Narwhals ZDTypes.

This module provides base classes and mixins that eliminate boilerplate
across ZDType implementations while preserving behavior.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import narwhals as nw
import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat

__all__ = [
    "SimpleZDType",
    "ZarrV3OnlyMixin",
]


class ZarrV3OnlyMixin:
    """Mixin for Zarr v3-only types with configuration.

    This mixin provides the v2 rejection methods and blocks auto-inference from_native_dtype.
    """

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZarrV3OnlyMixin:
        """Zarr v2 not supported."""
        msg = f"{cls.__name__} only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZarrV3OnlyMixin:
        """Prevent auto-inference to avoid registry conflicts."""
        msg = (
            f"{cls.__name__} cannot be inferred from numpy dtype {dtype}. "
            f"Use explicit construction. "
            f"This prevents registry conflicts with standard dtypes."
        )
        raise DataTypeValidationError(msg)


@dataclass(frozen=True)
class SimpleZDType(ZDType):
    """Base class for simple (non-configurable) Narwhals ZDTypes.

    This base class handles all the common boilerplate for simple types that:
    - Have no configuration fields
    - Use empty configuration in JSON serialization
    - Block auto-inference from numpy dtypes

    Subclasses must define:
    - _zarr_v3_name: ClassVar[str] - The Zarr v3 type name (e.g., "narwhals.int8")
    - dtype_cls: ClassVar[type] - The numpy dtype class (e.g., np.int8)
    - nw_dtype: property - Returns the corresponding Narwhals dtype

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class ZNarwhalsInt8(SimpleZDType):
    ...     _zarr_v3_name: ClassVar[str] = "narwhals.int8"
    ...     dtype_cls: ClassVar[type] = np.int8
    ...
    ...     @property
    ...     def nw_dtype(self) -> nw.DType:
    ...         return nw.Int8
    """

    _zarr_v3_name: ClassVar[str]
    dtype_cls: ClassVar[type]

    @property
    @abstractmethod
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        ...

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"{self.__class__.__name__} only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)
        return {"name": self._zarr_v3_name, "configuration": {}}

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format."""
        if not isinstance(data, dict):
            return False
        return data.get("name") == cls._zarr_v3_name

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> SimpleZDType:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)
        return cls()

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> SimpleZDType:
        """Zarr v2 not supported."""
        msg = f"{cls.__name__} only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> SimpleZDType:
        """Prevent auto-inference to avoid registry conflicts."""
        msg = (
            f"{cls.__name__} cannot be inferred from numpy dtype {dtype}. "
            f"Use explicit construction: {cls.__name__}(). "
            f"This prevents registry conflicts with standard dtypes."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy dtype."""
        return np.dtype(self.dtype_cls)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid numeric scalar."""
        return isinstance(data, (int, np.integer, float, np.floating))

    def _cast_scalar_unchecked(self, data: int | float) -> np.generic:
        """Cast scalar to the dtype_cls type."""
        return self.dtype_cls(data)

    def cast_scalar(self, data: object) -> np.generic:
        """Cast object to scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to {self.dtype_cls.__name__}"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.generic:
        """Default scalar value (0)."""
        return self.dtype_cls(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.generic:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        val = self.cast_scalar(data)
        # Use float for floating-point types, int for integer types
        if np.issubdtype(self.dtype_cls, np.floating):
            return float(val)
        return int(val)
