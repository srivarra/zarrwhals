"""Float32 data type for Narwhals float32 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsFloat32(ZDType):
    """Custom Zarr v3 dtype for Narwhals Float32 (32-bit floating point).

    Stores single-precision floating point values with explicit type tracking.
    This preserves Float32 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsFloat32
    >>> dtype = ZNarwhalsFloat32()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.float32', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('float32')

    Notes
    -----
    - Registered as "narwhals.float32" (ZEP0009 compliant)
    - Stores as native float32
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.float32"
    dtype_cls: ClassVar[type] = np.float32

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsFloat32 only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsFloat32:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsFloat32:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsFloat32 only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsFloat32:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsFloat32 cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsFloat32(). "
            "This prevents registry conflicts with standard Float32 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy float32 dtype."""
        return np.dtype("float32")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid float32 scalar."""
        return isinstance(data, (int, float, np.integer, np.floating))

    def _cast_scalar_unchecked(self, data: int | float | np.floating) -> np.float32:
        """Cast scalar to float32."""
        return np.float32(data)

    def cast_scalar(self, data: object) -> np.float32:
        """Cast object to float32 scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to float32"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.float32:
        """Default float32 scalar (0.0)."""
        return np.float32(0.0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.float32:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return float(self.cast_scalar(data))
