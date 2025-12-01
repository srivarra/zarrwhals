"""Float64 data type for Narwhals float64 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsFloat64(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Float64 (64-bit floating point).

    Stores double-precision floating point values with explicit type tracking.
    This preserves Float64 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsFloat64
    >>> dtype = ZNarwhalsFloat64()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.float64', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('float64')

    Notes
    -----
    - Registered as "narwhals.float64" (ZEP0009 compliant)
    - Stores as native float64
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.float64"
    dtype_cls: ClassVar[type] = np.float64

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Float64
