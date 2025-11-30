"""Struct data type for Narwhals struct encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsStruct(ZDType):
    """Custom Zarr v3 dtype for Narwhals Struct (named field structures).

    Stores structs as JSON-serialized strings with field type information.
    This supports nested structures while preserving field names and types.

    Parameters
    ----------
    fields : tuple[tuple[str, str], ...]
        Tuple of (field_name, dtype_str) pairs. Uses tuple for frozen dataclass.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsStruct
    >>> dtype = ZNarwhalsStruct(fields=(("a", "Int64"), ("b", "String")))
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.struct', 'configuration': {'fields': [{'name': 'a', 'dtype': 'Int64'}, {'name': 'b', 'dtype': 'String'}]}}

    Notes
    -----
    - Registered as "narwhals.struct" (ZEP0009 compliant)
    - Stores as VariableLengthUTF8 (JSON strings)
    - Field names and types preserved for reconstruction
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.struct"
    dtype_cls: ClassVar[type] = np.object_  # JSON-serialized strings

    fields: tuple[tuple[str, str], ...] = ()  # ((name, dtype_str), ...)

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsStruct only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"fields": [{"name": n, "dtype": d} for n, d in self.fields]},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with fields."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "fields" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsStruct:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        fields_list = config["fields"]
        fields = tuple((f["name"], f["dtype"]) for f in fields_list)

        return cls(fields=fields)

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsStruct:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsStruct only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsStruct:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsStruct cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsStruct(fields=((name, dtype), ...)). "
            "This prevents registry conflicts with standard Object dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy object dtype (for JSON strings)."""
        return np.dtype(object)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid struct scalar."""
        return isinstance(data, (dict, str))  # dict or JSON string

    def _cast_scalar_unchecked(self, data: dict | str) -> str:
        """Cast scalar to JSON string."""
        import json

        if isinstance(data, str):
            return data  # Already JSON string
        return json.dumps(data)

    def cast_scalar(self, data: object) -> str:
        """Cast object to struct scalar (JSON string)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to struct"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> str:
        """Default struct scalar (empty object as JSON)."""
        return "{}"

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """Deserialize scalar from JSON."""
        import json

        if isinstance(data, str):
            return data  # Already JSON string
        return json.dumps(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return self.cast_scalar(data)
