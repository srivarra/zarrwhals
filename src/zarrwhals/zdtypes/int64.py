"""Int64 data type for Narwhals int64 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsInt64(ZDType):
    """Custom Zarr v3 dtype for Narwhals Int64 (64-bit signed integer).

    Stores 64-bit signed integer values with explicit type tracking.
    This preserves Int64 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt64
    >>> dtype = ZNarwhalsInt64()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int64', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int64')

    Notes
    -----
    - Registered as "narwhals.int64" (ZEP0009 compliant)
    - Stores as native int64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int64"
    dtype_cls: ClassVar[type] = np.int64

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsInt64 only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsInt64:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsInt64:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsInt64 only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsInt64:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsInt64 cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsInt64(). "
            "This prevents registry conflicts with standard Int64 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy int64 dtype."""
        return np.dtype("int64")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid int64 scalar."""
        return isinstance(data, (int, np.integer))

    def _cast_scalar_unchecked(self, data: int | np.integer) -> np.int64:
        """Cast scalar to int64."""
        return np.int64(data)

    def cast_scalar(self, data: object) -> np.int64:
        """Cast object to int64 scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to int64"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int64:
        """Default int64 scalar (0)."""
        return np.int64(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
