"""UInt16 data type for Narwhals uint16 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsUInt16(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals UInt16 (16-bit unsigned integer).

    Stores 16-bit unsigned integer values with explicit type tracking.
    This preserves UInt16 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsUInt16
    >>> dtype = ZNarwhalsUInt16()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.uint16', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('uint16')

    Notes
    -----
    - Registered as "narwhals.uint16" (ZEP0009 compliant)
    - Stores as native uint16
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.uint16"
    dtype_cls: ClassVar[type] = np.uint16

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.UInt16
