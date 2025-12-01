"""List data type for Narwhals list encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import narwhals as nw
import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

from .base import ZarrV3OnlyMixin

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsList(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals List (variable-length lists).

    Stores lists as JSON-serialized strings with inner dtype information.
    This supports variable-length lists per element while preserving type info.

    Parameters
    ----------
    inner_dtype : str
        String representation of the inner Narwhals dtype (e.g., "Int64", "String")

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsList
    >>> dtype = ZNarwhalsList(inner_dtype="Int64")
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.list', 'configuration': {'inner_dtype': 'Int64'}}

    Notes
    -----
    - Registered as "narwhals.list" (ZEP0009 compliant)
    - Stores as VariableLengthUTF8 (JSON strings)
    - Inner dtype preserved for reconstruction
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.list"
    dtype_cls: ClassVar[type] = np.object_  # JSON-serialized strings

    inner_dtype: str = "String"

    @property
    def nw_dtype(self) -> nw.List:
        """Return corresponding Narwhals dtype."""
        from .converters import _parse_inner_dtype

        return nw.List(_parse_inner_dtype(self.inner_dtype))

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsList only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"inner_dtype": self.inner_dtype},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with inner_dtype."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "inner_dtype" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsList:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        inner_dtype = config["inner_dtype"]

        return cls(inner_dtype=inner_dtype)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy object dtype (for JSON strings)."""
        return np.dtype(object)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid list scalar."""
        return isinstance(data, (list, str))  # list or JSON string

    def _cast_scalar_unchecked(self, data: list | str) -> str:
        """Cast scalar to JSON string."""
        import json

        if isinstance(data, str):
            return data  # Already JSON string
        return json.dumps(data)

    def cast_scalar(self, data: object) -> str:
        """Cast object to list scalar (JSON string)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to list"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> str:
        """Default list scalar (empty list as JSON)."""
        return "[]"

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """Deserialize scalar from JSON."""
        import json

        if isinstance(data, str):
            return data  # Already JSON string
        return json.dumps(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return self.cast_scalar(data)
