"""Datetime data type for Narwhals datetime encoding in Zarr v3."""

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
class ZNarwhalsDatetime(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals Datetime with timezone support.

    Extends Zarr's built-in DateTime64 to add timezone information, which is critical
    for proper datetime handling in Narwhals (nw.Datetime supports time_zone parameter).

    Datetimes are stored as int64 (nanoseconds/microseconds/milliseconds/seconds since epoch)
    with timezone metadata preserved.

    Parameters
    ----------
    time_unit : {"ns", "us", "ms", "s"}, default "ns"
        Time resolution unit
    time_zone : str | None, default None
        IANA timezone string (e.g., "UTC", "America/New_York") or None for timezone-naive

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZDatetime
    >>> # Timezone-aware datetime
    >>> dtype = ZDatetime(time_unit="ns", time_zone="UTC")
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.datetime', 'configuration': {'time_unit': 'ns', 'time_zone': 'UTC'}}

    >>> # Timezone-naive datetime
    >>> dtype = ZDatetime(time_unit="ms", time_zone=None)
    >>> dtype.to_native_dtype()
    dtype('<M8[ms]')

    Notes
    -----
    - Registered as "narwhals.datetime" (ZEP0009 compliant)
    - Stores as int64 view of datetime64 array
    - Timezone preserved in configuration (Zarr's DateTime64 doesn't support timezone)
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.datetime"
    dtype_cls: ClassVar[type] = np.int64  # Stored as int64 view

    time_unit: Literal["ns", "us", "ms", "s"] = "ns"
    time_zone: str | None = None

    @property
    def nw_dtype(self) -> nw.Datetime:
        """Return corresponding Narwhals dtype."""
        kwargs: dict = {"time_unit": self.time_unit}
        if self.time_zone:
            kwargs["time_zone"] = self.time_zone
        return nw.Datetime(**kwargs)

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZDatetime only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {
                "time_unit": self.time_unit,
                "time_zone": self.time_zone,
            },
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
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsDatetime:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        time_unit = config["time_unit"]
        time_zone = config.get("time_zone")

        if time_unit not in ("ns", "us", "ms", "s"):
            msg = f"Invalid time_unit '{time_unit}', must be one of: ns, us, ms, s"
            raise DataTypeValidationError(msg)

        return cls(time_unit=time_unit, time_zone=time_zone)  # type: ignore[arg-type]

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy datetime64 dtype with time unit."""
        return np.dtype(f"datetime64[{self.time_unit}]")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid datetime scalar."""
        return isinstance(data, (int, np.integer, np.datetime64))

    def _cast_scalar_unchecked(self, data: int | np.datetime64) -> np.int64:
        """Cast scalar to int64 (datetime stored as int64 view)."""
        if isinstance(data, np.datetime64):
            return np.int64(data.view(np.int64))
        return np.int64(data)

    def cast_scalar(self, data: object) -> np.int64:
        """Cast object to datetime scalar (int64 view)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to datetime"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int64:
        """Default datetime scalar (epoch)."""
        return np.int64(0)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64:
        """Deserialize scalar from JSON."""
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return int(self.cast_scalar(data))
