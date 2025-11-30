"""Int32 data type for Narwhals int32 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsInt32(ZDType):
    """Custom Zarr v3 dtype for Narwhals Int32 (32-bit signed integer).

    Stores 32-bit signed integer values with explicit type tracking.
    This preserves Int32 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt32
    >>> dtype = ZNarwhalsInt32()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int32', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int32')

    Notes
    -----
    - Registered as "narwhals.int32" (ZEP0009 compliant)
    - Stores as native int32
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int32"
    dtype_cls: ClassVar[type] = np.int32

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsInt32 only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsInt32:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsInt32:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsInt32 only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsInt32:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsInt32 cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsInt32(). "
            "This prevents registry conflicts with standard Int32 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy int32 dtype."""
        return np.dtype("int32")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid int32 scalar."""
        return isinstance(data, (int, np.integer))

    def _cast_scalar_unchecked(self, data: int | np.integer) -> np.int32:
        """Cast scalar to int32."""
        return np.int32(data)

    def cast_scalar(self, data: object) -> np.int32:
        """Cast object to int32 scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to int32"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int32:
        """Default int32 scalar (0)."""
        return np.int32(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
