"""Time data type for Narwhals time encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, TypeGuard

import narwhals as nw
import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

from .base import ZarrV3OnlyMixin

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsTime(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals Time (time-of-day without date).

    Stores time as int64 representing time since midnight in the specified unit.
    This preserves time resolution without date information.

    Parameters
    ----------
    time_unit : {"ns", "us", "ms", "s"}, default "ns"
        Time resolution unit

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsTime
    >>> dtype = ZNarwhalsTime(time_unit="ns")
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.time', 'configuration': {'time_unit': 'ns'}}

    Notes
    -----
    - Registered as "narwhals.time" (ZEP0009 compliant)
    - Stores as int64 (time since midnight in specified unit)
    - No datetime64 equivalent; remains as int64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.time"
    dtype_cls: ClassVar[type] = np.int64  # Time since midnight

    time_unit: Literal["ns", "us", "ms", "s"] = "ns"

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Time

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsTime only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"time_unit": self.time_unit},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with time_unit."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "time_unit" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsTime:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        time_unit = config["time_unit"]

        if time_unit not in ("ns", "us", "ms", "s"):
            msg = f"Invalid time_unit '{time_unit}', must be one of: ns, us, ms, s"
            raise DataTypeValidationError(msg)

        return cls(time_unit=time_unit)  # type: ignore[arg-type]

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy int64 dtype (no datetime64 equivalent for time-only)."""
        return np.dtype(np.int64)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid time scalar."""
        return isinstance(data, (int, np.integer))

    def _cast_scalar_unchecked(self, data: int) -> np.int64:
        """Cast scalar to int64."""
        return np.int64(data)

    def cast_scalar(self, data: object) -> np.int64:
        """Cast object to time scalar (int64)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to time"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int64:
        """Default time scalar (midnight)."""
        return np.int64(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
