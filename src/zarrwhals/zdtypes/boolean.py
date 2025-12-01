"""Boolean data type for Narwhals boolean encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import narwhals as nw
import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

from .base import ZarrV3OnlyMixin

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsBoolean(ZarrV3OnlyMixin, ZDType):
    """Custom Zarr v3 dtype for Narwhals Boolean.

    Stores boolean values with explicit type tracking.
    This preserves Boolean semantics through Zarr storage round-trips.

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsBoolean
    >>> dtype = ZNarwhalsBoolean()
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.boolean', 'configuration': {}}

    >>> dtype.to_native_dtype()
    dtype('bool')

    Notes
    -----
    - Registered as "narwhals.boolean" (ZEP0009 compliant)
    - Stores as native bool
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.boolean"
    dtype_cls: ClassVar[type] = np.bool_

    @property
    def nw_dtype(self) -> nw.DType:
        """Return corresponding Narwhals dtype."""
        return nw.Boolean

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsBoolean only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format."""
        if not isinstance(data, dict):
            return False
        return data.get("name") == cls._zarr_v3_name

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsBoolean:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)
        return cls()

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy bool dtype."""
        return np.dtype("bool")

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid boolean scalar."""
        return isinstance(data, (bool, np.bool_))

    def _cast_scalar_unchecked(self, data: bool | np.bool_) -> np.bool_:
        """Cast scalar to bool."""
        return np.bool_(data)

    def cast_scalar(self, data: object) -> np.bool_:
        """Cast object to boolean scalar."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to bool"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.bool_:
        """Default boolean scalar (False)."""
        return np.bool_(False)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.bool_:
        """Deserialize scalar from JSON."""
        return np.bool_(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        return bool(self.cast_scalar(data))
