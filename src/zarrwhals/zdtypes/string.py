"""String data type for Narwhals string encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import narwhals as nw
import numpy as np
from numpy.dtypes import StringDType
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

from .base import ZarrV3OnlyMixin

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsString(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals String (variable-length UTF-8).

    Stores variable-length UTF-8 strings with explicit type tracking.
    This preserves String semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsString
    >>> dtype = ZNarwhalsString()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.string', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('StringDType')

    Notes
    -----
    - Registered as "narwhals.string" (ZEP0009 compliant)
    - Stores as NumPy StringDType (variable-length UTF-8)
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.string"
    dtype_cls: ClassVar[type] = str

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.String

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsString only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsString:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)
        return cls()

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy StringDType."""
        return np.dtype(StringDType())

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid string scalar."""
        return isinstance(data, str)

    def _cast_scalar_unchecked(self, data: str) -> str:
        """Cast scalar to string."""
        return str(data)

    def cast_scalar(self, data: object) -> str:
        """Cast object to string scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to string"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> str:
        """Default string scalar (empty string)."""
        return ""

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """Deserialize scalar from JSON."""
        return str(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return self.cast_scalar(data)
