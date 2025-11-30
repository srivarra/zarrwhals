"""Int8 data type for Narwhals int8 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsInt8(ZDType):
    """Custom Zarr v3 dtype for Narwhals Int8 (8-bit signed integer).

    Stores 8-bit signed integer values with explicit type tracking.
    This preserves Int8 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt8
    >>> dtype = ZNarwhalsInt8()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int8', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int8')

    Notes
    -----
    - Registered as "narwhals.int8" (ZEP0009 compliant)
    - Stores as native int8
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int8"
    dtype_cls: ClassVar[type] = np.int8

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsInt8 only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsInt8:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsInt8:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsInt8 only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsInt8:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsInt8 cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsInt8(). "
            "This prevents registry conflicts with standard Int8 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy int8 dtype."""
        return np.dtype("int8")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid int8 scalar."""
        return isinstance(data, (int, np.integer))

    def _cast_scalar_unchecked(self, data: int | np.integer) -> np.int8:
        """Cast scalar to int8."""
        return np.int8(data)

    def cast_scalar(self, data: object) -> np.int8:
        """Cast object to int8 scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to int8"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int8:
        """Default int8 scalar (0)."""
        return np.int8(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int8:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
