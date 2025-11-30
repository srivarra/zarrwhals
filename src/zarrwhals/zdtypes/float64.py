"""Float64 data type for Narwhals float64 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsFloat64(ZDType):
    """Custom Zarr v3 dtype for Narwhals Float64 (64-bit floating point).

    Stores double-precision floating point values with explicit type tracking.
    This preserves Float64 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsFloat64
    >>> dtype = ZNarwhalsFloat64()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.float64', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('float64')

    Notes
    -----
    - Registered as "narwhals.float64" (ZEP0009 compliant)
    - Stores as native float64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.float64"
    dtype_cls: ClassVar[type] = np.float64

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsFloat64 only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format."""
        if not isinstance(data, dict):
            return False
        return data.get("name") == cls._zarr_v3_name

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsFloat64:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsFloat64:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsFloat64 only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsFloat64:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsFloat64 cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsFloat64(). "
            "This prevents registry conflicts with standard Float64 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy float64 dtype."""
        return np.dtype("float64")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid float64 scalar."""
        return isinstance(data, (int, float, np.integer, np.floating))

    def _cast_scalar_unchecked(self, data: int | float | np.floating) -> np.float64:
        """Cast scalar to float64."""
        return np.float64(data)

    def cast_scalar(self, data: object) -> np.float64:
        """Cast object to float64 scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to float64"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.float64:
        """Default float64 scalar (0.0)."""
        return np.float64(0.0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float64:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return float(self.cast_scalar(data))
