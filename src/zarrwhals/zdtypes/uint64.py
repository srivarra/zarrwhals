"""UInt64 data type for Narwhals uint64 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsUInt64(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals UInt64 (64-bit unsigned integer).

    Stores 64-bit unsigned integer values with explicit type tracking.
    This preserves UInt64 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsUInt64
    >>> dtype = ZNarwhalsUInt64()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.uint64', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('uint64')

    Notes
    -----
    - Registered as "narwhals.uint64" (ZEP0009 compliant)
    - Stores as native uint64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.uint64"
    dtype_cls: ClassVar[type] = np.uint64

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.UInt64
