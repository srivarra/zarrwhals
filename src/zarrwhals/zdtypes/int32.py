"""Int32 data type for Narwhals int32 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsInt32(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Int32 (32-bit signed integer).

    Stores 32-bit signed integer values with explicit type tracking.
    This preserves Int32 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt32
    >>> dtype = ZNarwhalsInt32()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int32', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int32')

    Notes
    -----
    - Registered as "narwhals.int32" (ZEP0009 compliant)
    - Stores as native int32
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int32"
    dtype_cls: ClassVar[type] = np.int32

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Int32
