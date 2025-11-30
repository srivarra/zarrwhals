"""Array data type for Narwhals fixed-shape array encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsArray(ZDType):
    """Custom Zarr v3 dtype for Narwhals Array (fixed-shape arrays per row).

    Stores fixed-shape arrays as multidimensional arrays with inner dtype info.
    Unlike List, Array has a fixed shape known at dtype definition time.

    Parameters
    ----------
    inner_dtype : str
        String representation of the inner Narwhals dtype (e.g., "Float32")
    shape : tuple[int, ...]
        Fixed shape of each array element

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZNarwhalsArray
    >>> dtype = ZNarwhalsArray(inner_dtype="Float32", shape=(3,))
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.array', 'configuration': {'inner_dtype': 'Float32', 'shape': [3]}}

    Notes
    -----
    - Registered as "narwhals.array" (ZEP0009 compliant)
    - Stores as native multidimensional arrays
    - Shape is fixed per dtype definition
    - Zarr v3 only
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.array"
    dtype_cls: ClassVar[type] = np.ndarray

    inner_dtype: str = "Float64"
    shape: tuple[int, ...] = (1,)

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize to Zarr v3 JSON format."""
        if zarr_format != 3:
            msg = f"ZNarwhalsArray only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {
                "inner_dtype": self.inner_dtype,
                "shape": list(self.shape),  # Convert tuple to list for JSON
            },
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with inner_dtype and shape."""
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        config = data.get("configuration", {})
        return "inner_dtype" in config and "shape" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsArray:
        """Deserialize from Zarr v3 JSON."""
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        config = data["configuration"]  # type: ignore[index]
        inner_dtype = config["inner_dtype"]
        shape = tuple(config["shape"])

        return cls(inner_dtype=inner_dtype, shape=shape)

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported."""
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsArray:
        """Zarr v2 not supported."""
        msg = "ZNarwhalsArray only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsArray:
        """Prevent auto-inference to avoid conflicts."""
        msg = (
            f"ZNarwhalsArray cannot be inferred from numpy dtype {dtype}. "
            "Use explicit construction: ZNarwhalsArray(inner_dtype='Float32', shape=(3,)). "
            "This prevents registry conflicts."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy dtype based on inner_dtype."""
        # Map common Narwhals dtype strings to NumPy dtypes
        dtype_map = {
            "Int8": np.int8,
            "Int16": np.int16,
            "Int32": np.int32,
            "Int64": np.int64,
            "UInt8": np.uint8,
            "UInt16": np.uint16,
            "UInt32": np.uint32,
            "UInt64": np.uint64,
            "Float32": np.float32,
            "Float64": np.float64,
            "Boolean": np.bool_,
        }
        np_dtype = dtype_map.get(self.inner_dtype, np.float64)
        return np.dtype(np_dtype)

    def _check_scalar(self, data: object) -> bool:
        """Check if data is valid array scalar."""
        return isinstance(data, (list, np.ndarray))

    def _cast_scalar_unchecked(self, data: list | np.ndarray) -> np.ndarray:
        """Cast scalar to numpy array."""
        arr = np.asarray(data, dtype=self.to_native_dtype())
        if arr.shape != self.shape:
            msg = f"Array shape {arr.shape} does not match expected shape {self.shape}"
            raise ValueError(msg)
        return arr

    def cast_scalar(self, data: object) -> np.ndarray:
        """Cast object to array scalar (numpy array)."""
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to array"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.ndarray:
        """Default array scalar (zeros)."""
        return np.zeros(self.shape, dtype=self.to_native_dtype())

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.ndarray:
        """Deserialize scalar from JSON."""
        if isinstance(data, list):
            return self.cast_scalar(data)
        msg = f"Cannot deserialize {type(data)} to array"
        raise TypeError(msg)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON."""
        arr = self.cast_scalar(data)
        return arr.tolist()
