"""Object data type for Narwhals object encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsObject(ZDType):
    """Custom Zarr v3 dtype for Narwhals Object (arbitrary Python objects).

    Stores objects as JSON-serialized strings. Useful for untyped or
    heterogeneous data that doesn't fit other type categories.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsObject
    >>> dtype = ZNarwhalsObject()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.object', 'configuration': {}}

    Notes
    -----
    - Registered as "narwhals.object" (ZEP0009 compliant)
    - Stores as VariableLengthUTF8 (JSON strings)
    - Fallback for untyped data
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.object"
    dtype_cls: ClassVar[type] = np.object_  # JSON-serialized strings

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsObject only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsObject:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsObject:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsObject only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsObject:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsObject cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsObject(). "
            "This prevents registry conflicts with standard Object dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy object dtype."""
        return np.dtype(object)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid object scalar (any JSON-serializable)."""
        return True  # Accept anything, will serialize to JSON

    def _cast_scalar_unchecked(self, data: object) -> str:
        """Cast scalar to JSON string."""
        import json

        return json.dumps(data, default=str)

    def cast_scalar(self, data: object) -> str:
        """Cast object to object scalar (JSON string)."""
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> str:
        """Default object scalar (null as JSON)."""
        return "null"

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """Deserialize scalar from JSON."""
        import json

        if isinstance(data, str):
            return data  # Already JSON string
        return json.dumps(data, default=str)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return self.cast_scalar(data)
