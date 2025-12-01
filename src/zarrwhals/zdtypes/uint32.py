"""UInt32 data type for Narwhals uint32 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsUInt32(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals UInt32 (32-bit unsigned integer).

    Stores 32-bit unsigned integer values with explicit type tracking.
    This preserves UInt32 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsUInt32
    >>> dtype = ZNarwhalsUInt32()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.uint32', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('uint32')

    Notes
    -----
    - Registered as "narwhals.uint32" (ZEP0009 compliant)
    - Stores as native uint32
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.uint32"
    dtype_cls: ClassVar[type] = np.uint32

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.UInt32
