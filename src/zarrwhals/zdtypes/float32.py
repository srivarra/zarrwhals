"""Float32 data type for Narwhals float32 encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import narwhals as nw
import numpy as np

from .base import SimpleZDType


@dataclass(frozen=True)
class ZNarwhalsFloat32(SimpleZDType):
    """Custom Zarr v3 dtype for Narwhals Float32 (32-bit floating point).

    Stores single-precision floating point values with explicit type tracking.
    This preserves Float32 semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsFloat32
    >>> dtype = ZNarwhalsFloat32()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.float32', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('float32')

    Notes
    -----
    - Registered as "narwhals.float32" (ZEP0009 compliant)
    - Stores as native float32
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.float32"
    dtype_cls: ClassVar[type] = np.float32

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Float32
