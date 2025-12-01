"""Int8 data type for Narwhals int8 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsInt8(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Int8 (8-bit signed integer).

    Stores 8-bit signed integer values with explicit type tracking.
    This preserves Int8 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsInt8
    >>> dtype = ZNarwhalsInt8()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.int8', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('int8')

    Notes
    -----
    - Registered as "narwhals.int8" (ZEP0009 compliant)
    - Stores as native int8
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.int8"
    dtype_cls: ClassVar[type] = np.int8

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Int8
