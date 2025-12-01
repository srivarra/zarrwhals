"""UInt8 data type for Narwhals uint8 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsUInt8(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals UInt8 (8-bit unsigned integer).

    Stores 8-bit unsigned integer values with explicit type tracking.
    This preserves UInt8 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsUInt8
    >>> dtype = ZNarwhalsUInt8()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.uint8', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('uint8')

    Notes
    -----
    - Registered as "narwhals.uint8" (ZEP0009 compliant)
    - Stores as native uint8
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.uint8"
    dtype_cls: ClassVar[type] = np.uint8

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.UInt8
