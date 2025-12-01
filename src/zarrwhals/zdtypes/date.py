"""Date data type for Narwhals date encoding in Zarr v3."""

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
class ZNarwhalsDate(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals Date (calendar date without time).

    Stores dates as int32 representing days since Unix epoch (1970-01-01).
    This provides efficient storage while preserving date semantics.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsDate
    >>> dtype = ZNarwhalsDate()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.date', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('<M8[D]')

    Notes
    -----
    - Registered as "narwhals.date" (ZEP0009 compliant)
    - Stores as int32 (days since epoch), viewed as datetime64[D]
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.date"
    dtype_cls: ClassVar[type] = np.int32  # Days since epoch

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Date

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsDate only supports Zarr v3, got format {zarr_format}"
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsDate:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)
        return cls()

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy datetime64[D] dtype."""
        return np.dtype("datetime64[D]")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid date scalar."""
        return isinstance(data, (int, np.integer, np.datetime64))

    def _cast_scalar_unchecked(self, data: int | np.datetime64) -> np.int32:
        """Cast scalar to int32 (days since epoch)."""
        if isinstance(data, np.datetime64):
            # Convert to days since epoch
            return np.int32(data.astype("datetime64[D]").view(np.int64))
        return np.int32(data)

    def cast_scalar(self, data: object) -> np.int32:
        """Cast object to date scalar (int32 days)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to date"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int32:
        """Default date scalar (epoch: 1970-01-01)."""
        return np.int32(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
