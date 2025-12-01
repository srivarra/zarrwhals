"""Int16 data type for Narwhals int16 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsInt16(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Int16 (16-bit signed integer).

    Stores 16-bit signed integer values with explicit type tracking.
    This preserves Int16 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt16
    >>> dtype = ZNarwhalsInt16()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int16', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int16')

    Notes
    -----
    - Registered as "narwhals.int16" (ZEP0009 compliant)
    - Stores as native int16
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int16"
    dtype_cls: ClassVar[type] = np.int16

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Int16
