"""Duration data type for Narwhals duration encoding in Zarr v3."""

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
class ZNarwhalsDuration(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals Duration (time deltas).

    Similar to Zarr's TimeDelta64 but with Narwhals-compatible naming (nw.Duration).
    Durations are stored as int64 (nanoseconds/microseconds/milliseconds/seconds).

    Parameters
    ----------
    time_unit : {"ns", "us", "ms", "s"}, default "ns"
        Time resolution unit for the duration

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZDuration
    >>> dtype = ZDuration(time_unit="ms")
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.duration', 'configuration': {'time_unit': 'ms'}}

    >>> dtype.to_native_dtype()
    dtype('<m8[ms]')

    Notes
    -----
    - Registered as "narwhals.duration" (ZEP0009 compliant)
    - Stores as int64 view of timedelta64 array
    - Equivalent to Zarr's TimeDelta64 with Narwhals naming
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.duration"
    dtype_cls: ClassVar[type] = np.int64

    time_unit: Literal["ns", "us", "ms", "s"] = "ns"

    @property
    def nw_dtype(self) -> nw.Duration:
        """Return corresponding Narwhals dtype."""
        return nw.Duration(time_unit=self.time_unit)

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZDuration only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"time_unit": self.time_unit},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "time_unit" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsDuration:
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
        """Convert to NumPy timedelta64 dtype with time unit."""
        return np.dtype(f"timedelta64[{self.time_unit}]")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid duration scalar."""
        return isinstance(data, (int, np.integer, np.timedelta64))

    def _cast_scalar_unchecked(self, data: int | np.timedelta64) -> np.int64:
        """Cast scalar to int64 (duration stored as int64 view)."""
        if isinstance(data, np.timedelta64):
            return np.int64(data.view(np.int64))
        return np.int64(data)

    def cast_scalar(self, data: object) -> np.int64:
        """Cast object to duration scalar (int64 view)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to duration"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int64:
        """Default duration scalar (zero duration)."""
        return np.int64(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
