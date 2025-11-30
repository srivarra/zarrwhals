"""Enum data type for Narwhals enum encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsEnum(ZDType):
    """Custom Zarr v3 dtype for Narwhals Enum (ordered categorical with fixed categories).

    Unlike Categorical (dynamic categories from data), Enum has predefined categories
    from the dtype definition. This matches Polars Enum semantics.

    Note: Enum data is stored as a Group (codes + categories arrays), not a single array.
    This ZEnum dtype provides metadata and type safety but isn't used directly for array creation.

    Parameters
    ----------
    categories : tuple
        Fixed, immutable tuple of category values (hashable for frozen dataclass)
    ordered : bool, default True
        Whether the enum supports ordering (typically True for enums)

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZEnum
    >>> dtype = ZEnum(categories=("small", "medium", "large"), ordered=True)
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.enum', 'configuration': {'categories': ['small', 'medium', 'large'], 'ordered': True}}

    Notes
    -----
    - Registered as "narwhals.enum" (ZEP0009 compliant)
    - Categories are fixed at dtype creation (unlike Categorical)
    - Typically ordered (enum semantics imply ordering)
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.enum"
    dtype_cls: ClassVar[type] = np.int32

    categories: tuple = ()
    ordered: bool = True

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZEnum only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {
                "categories": list(self.categories),  # Convert tuple to list for JSON
                "ordered": self.ordered,
            },
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with categories."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "categories" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsEnum:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        categories = tuple(config["categories"])  # Convert list to tuple
        ordered = config.get("ordered", True)

        return cls(categories=categories, ordered=ordered)

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsEnum:
        """Zarr v2 not supported."""
        msg = "ZEnum only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsEnum:
        """Prevent auto-inference."""
        msg = (
            f"ZEnum cannot be inferred from numpy dtype {dtype}. "
            f"Use explicit construction: ZEnum(categories=(...), ordered=True). "
            f"This prevents registry conflicts with standard Int32 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy dtype (int32 for enum codes)."""
        return np.dtype(np.int32)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid enum code scalar."""
        return isinstance(data, (int, np.integer))

    def _cast_scalar_unchecked(self, data: int) -> np.int32:
        """Cast scalar to int32 enum code."""
        return np.int32(data)

    def cast_scalar(self, data: object) -> np.int32:
        """Cast object to enum code scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to enum code (int32)"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int32:
        """Default enum code (-1 for missing)."""
        return np.int32(-1)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
