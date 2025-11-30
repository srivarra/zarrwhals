"""Binary data type for Narwhals binary encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsBinary(ZDType):
    """Custom Zarr v3 dtype for Narwhals Binary (raw bytes).

    Stores binary data as object arrays containing bytes objects.
    This supports variable-length binary data per element.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsBinary
    >>> dtype = ZNarwhalsBinary()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.binary', 'configuration': {}}

    Notes
    -----
    - Registered as "narwhals.binary" (ZEP0009 compliant)
    - Stores as object array containing bytes
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.binary"
    dtype_cls: ClassVar[type] = np.object_  # Variable-length bytes

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsBinary only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsBinary:
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
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsBinary:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsBinary only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsBinary:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsBinary cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsBinary(). "
            "This prevents registry conflicts with standard Object dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy object dtype."""
        return np.dtype(object)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid binary scalar."""
        return isinstance(data, (bytes, bytearray))

    def _cast_scalar_unchecked(self, data: bytes | bytearray) -> bytes:
        """Cast scalar to bytes."""
        return bytes(data)

    def cast_scalar(self, data: object) -> bytes:
        """Cast object to binary scalar (bytes)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to binary"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> bytes:
        """Default binary scalar (empty bytes)."""
        return b""

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> bytes:
        """Deserialize scalar from JSON (base64 encoded string)."""
        import base64

        if isinstance(data, str):
            return base64.b64decode(data)
        msg = f"Cannot deserialize {type(data)} to binary"
        raise TypeError(msg)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON (base64 encoded string)."""
        import base64

        return base64.b64encode(self.cast_scalar(data)).decode("ascii")
