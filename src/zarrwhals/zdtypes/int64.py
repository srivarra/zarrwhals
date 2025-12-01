"""Int64 data type for Narwhals int64 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsInt64(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Int64 (64-bit signed integer).

    Stores 64-bit signed integer values with explicit type tracking.
    This preserves Int64 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt64
    >>> dtype = ZNarwhalsInt64()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int64', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int64')

    Notes
    -----
    - Registered as "narwhals.int64" (ZEP0009 compliant)
    - Stores as native int64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int64"
    dtype_cls: ClassVar[type] = np.int64

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Int64
