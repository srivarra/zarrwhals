"""Decimal data type for Narwhals decimal encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsDecimal(ZDType):
    """Custom Zarr v3 dtype for Narwhals Decimal (arbitrary precision).

    Stores decimals as string representations to preserve full precision.
    Optionally includes precision and scale metadata.

    Parameters
    ----------
    precision : int | None, default None
        Total number of digits (if known)
    scale : int | None, default None
        Number of digits after decimal point (if known)

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsDecimal
    >>> dtype = ZNarwhalsDecimal(precision=10, scale=2)
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.decimal', 'configuration': {'precision': 10, 'scale': 2}}

    Notes
    -----
    - Registered as "narwhals.decimal" (ZEP0009 compliant)
    - Stores as VariableLengthUTF8 (string representation)
    - Precision/scale optional but preserved if provided
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.decimal"
    dtype_cls: ClassVar[type] = np.object_  # String representation

    precision: int | None = None
    scale: int | None = None

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsDecimal only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {
                "precision": self.precision,
                "scale": self.scale,
            },
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format."""
        if not isinstance(data, dict):
            return False
        return data.get("name") == cls._zarr_v3_name

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsDecimal:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data.get("configuration", {})  # type: ignore[union-attr]
        precision = config.get("precision")
        scale = config.get("scale")

        return cls(precision=precision, scale=scale)

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsDecimal:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsDecimal only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsDecimal:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsDecimal cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsDecimal(precision=10, scale=2). "
            "This prevents registry conflicts with standard Object dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy object dtype (for string representation)."""
        return np.dtype(object)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid decimal scalar."""
        from decimal import Decimal

        return isinstance(data, (str, int, float, Decimal))

    def _cast_scalar_unchecked(self, data: str | int | float) -> str:
        """Cast scalar to string representation."""
        return str(data)

    def cast_scalar(self, data: object) -> str:
        """Cast object to decimal scalar (string)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to decimal"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> str:
        """Default decimal scalar (zero)."""
        return "0"

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> str:
        """Deserialize scalar from JSON."""
        return str(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return self.cast_scalar(data)
