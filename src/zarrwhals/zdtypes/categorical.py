"""Categorical data types for Narwhals categorical encoding in Zarr v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, TypeGuard

import numpy as np
from zarr.core.dtype import DataTypeValidationError, DTypeJSON, ZDType

if TYPE_CHECKING:
    from zarr.core.common import JSON, ZarrFormat


@dataclass(frozen=True)
class ZNarwhalsCategoricalCodes(ZDType):
    """Custom Zarr v3 dtype for categorical codes array.

    Stores integer codes that index into the categories array. Codes range from

    0 to n_categories-1, with -1 representing missing values.
    This dtype uses dict JSON format to avoid conflicts with standard Int32:
    - Standard int32: `"int32"` (string format)
    - ZCategoricalCodes: `{"name": "...", "configuration": {...}}` (dict format)

    Parameters
    ----------
    ordered : bool, default False
        Whether the categorical supports ordering operations

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZCategoricalCodes
    >>> dtype = ZCategoricalCodes(ordered=True)
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.categorical.codes', 'configuration': {'ordered': True}}

    Notes
    -----
    - Registered as "narwhals.categorical.codes" (ZEP0009 compliant)
    - Always stores as int32 for memory efficiency
    - Dict JSON format prevents conflict with standard Int32 dtype
    """

    _zarr_v3_name: ClassVar[str] = "narwhals.categorical.codes"
    dtype_cls: ClassVar[type] = np.int32

    ordered: bool = False

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize dtype to Zarr v3 JSON format.

        Parameters
        ----------
        zarr_format : ZarrFormat
            Zarr format version (must be 3)

        Returns
        -------
        dict
            Dict with name and configuration for v3
        """
        if zarr_format != 3:
            msg = f"ZCategoricalCodes only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"ordered": self.ordered},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with configuration.

        This strict check prevents conflicts with standard Int32 dtype which uses
        string format "int32", not dict format.

        Parameters
        ----------
        data : str or dict
            JSON data to validate

        Returns
        -------
        bool
            True if data matches v3 categorical codes format (must be dict)
        """
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        return "configuration" in data

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsCategoricalCodes:
        """Deserialize from Zarr v3 JSON.

        Parameters
        ----------
        data : dict
            JSON dict with name and configuration

        Returns
        -------
        ZCategoricalCodes
            Deserialized categorical codes dtype instance

        Raises
        ------
        ValueError
            If JSON format is invalid
        """
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        # Extract configuration
        config = data.get("configuration", {})  # type: ignore[union-attr]
        ordered = config.get("ordered", False)

        return cls(ordered=ordered)

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported - always returns False.

        Parameters
        ----------
        _data : str or dict
            JSON data to validate (unused)

        Returns
        -------
        bool
            Always False (v2 not supported)
        """
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsCategoricalCodes:
        """Zarr v2 not supported - always raises error.

        Parameters
        ----------
        data : dict
            JSON dict with name

        Returns
        -------
        ZCategoricalCodes
            Deserialized categorical codes dtype instance

        Raises
        ------
        ValueError
            If JSON format is invalid
        """
        msg = "ZCategoricalCodes only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsCategoricalCodes:
        """Convert NumPy dtype to ZCategoricalCodes.

        This method should NOT be used for automatic dtype inference from regular arrays.
        It will raise an error to prevent conflicts with standard Int32 dtype.

        ZCategoricalCodes should only be created explicitly via the constructor
        with the desired ordered specification.

        Parameters
        ----------
        dtype : np.dtype
            NumPy dtype (int32)

        Returns
        -------
        ZCategoricalCodes
            Never - always raises error

        Raises
        ------
        DataTypeValidationError
            Always raised - this dtype requires explicit configuration via constructor
        """
        msg = (
            f"ZCategoricalCodes cannot be inferred from numpy dtype {dtype}. "
            f"Use explicit construction: ZCategoricalCodes(ordered=True|False). "
            f"This prevents registry conflicts with standard Int32 dtype."
        )
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy dtype.

        Returns
        -------
        np.dtype
            int32 dtype for categorical codes
        """
        return np.dtype(np.int32)

    def _check_scalar(self, data: object) -> TypeGuard[int | np.int32]:
        """Check if data is valid scalar for categorical codes.

        Parameters
        ----------
        data : object
            Data to check

        Returns
        -------
        bool
            True if data is int or np.int32
        """
        return isinstance(data, (int, np.int32, np.integer))

    def _cast_scalar_unchecked(self, data: int | np.int32) -> np.int32:
        """Cast scalar to np.int32 without validation.

        Parameters
        ----------
        data : int or np.int32
            Scalar value to cast

        Returns
        -------
        np.int32
            Casted scalar
        """
        return np.int32(data)

    def cast_scalar(self, data: object) -> np.int32:
        """Cast object to categorical code scalar.

        Parameters
        ----------
        data : object
            Object to cast

        Returns
        -------
        np.int32
            Categorical code

        Raises
        ------
        TypeError
            If data cannot be cast to int32
        """
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to categorical code (int32)"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int32:
        """Get default scalar value (missing = -1).

        Returns
        -------
        np.int32
            Default value of -1 (missing category code)
        """
        return np.int32(-1)

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int32:
        """Deserialize scalar from JSON.

        Parameters
        ----------
        data : JSON
            JSON data
        zarr_format : ZarrFormat
            Zarr format version

        Returns
        -------
        np.int32
            Deserialized scalar
        """
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON.

        Parameters
        ----------
        data : object
            Scalar value
        zarr_format : ZarrFormat
            Zarr format version

        Returns
        -------
        JSON
            JSON-serializable int
        """
        return int(self.cast_scalar(data))


@dataclass(frozen=True)
class ZNarwhalsCategoricalCategories(ZDType):
    """Custom Zarr v3 dtype for categorical categories array.

    Stores the unique category values with preserved dtype (int64, float64, or string).
    This allows categorical data to have integer categories (e.g., status codes 200, 404, 500)
    or float categories (e.g., temperature bins 98.6, 99.1, 100.2) without forced string conversion.

    Parameters
    ----------
    inner_dtype : {"int64", "float64", "string"}
        The actual dtype of category values

    Examples
    --------
    >>> from zarrwhals.zdtypes import ZCategoricalCategories
    >>> # Integer categories
    >>> dtype = ZCategoricalCategories(inner_dtype="int64")
    >>> dtype.to_native_dtype()
    dtype('int64')

    >>> # Float categories
    >>> dtype = ZCategoricalCategories(inner_dtype="float64")
    >>> dtype.to_json(zarr_format=3)
    {'name': 'narwhals.categorical.categories', 'configuration': {'inner_dtype': 'float64'}}

    Notes
    -----
    - Registered as "narwhals.categorical.categories" (ZEP0009 compliant)
    - Supports three inner dtypes: int64, float64, string
    - Dict JSON format prevents conflicts with standard dtypes
    - Dynamic dtype_cls property based on inner_dtype configuration
    """

    # Class-level constants
    _zarr_v3_name: ClassVar[str] = "narwhals.categorical.categories"

    # Instance fields
    inner_dtype: Literal["int64", "float64", "string"]

    @property
    def dtype_cls(self) -> type:
        """Dynamic dtype class based on inner_dtype.

        Returns
        -------
        type
            np.int64, np.float64, or StringDType class
        """
        if self.inner_dtype == "int64":
            return np.int64
        elif self.inner_dtype == "float64":
            return np.float64
        else:  # "string"
            from numpy.dtypes import StringDType

            return StringDType

    def to_json(self, zarr_format: ZarrFormat) -> dict:
        """Serialize dtype to Zarr v3 JSON format.

        Parameters
        ----------
        zarr_format : ZarrFormat
            Zarr format version (must be 3)

        Returns
        -------
        dict
            Dict with name and configuration for v3
        """
        if zarr_format != 3:
            msg = f"ZCategoricalCategories only supports Zarr v3, got format {zarr_format}"
            raise ValueError(msg)

        return {
            "name": self._zarr_v3_name,
            "configuration": {"inner_dtype": self.inner_dtype},
        }

    @classmethod
    def _check_json_v3(cls, data: DTypeJSON) -> TypeGuard[dict]:
        """Validate v3 JSON format - requires dict with inner_dtype.

        Parameters
        ----------
        data : str or dict
            JSON data to validate

        Returns
        -------
        bool
            True if data matches v3 categorical categories format
        """
        if not isinstance(data, dict):
            return False
        if data.get("name") != cls._zarr_v3_name:
            return False
        # Require configuration with inner_dtype
        config = data.get("configuration", {})
        return "inner_dtype" in config

    @classmethod
    def _from_json_v3(cls, data: DTypeJSON) -> ZNarwhalsCategoricalCategories:
        """Deserialize from Zarr v3 JSON.

        Parameters
        ----------
        data : dict
            JSON dict with name and configuration

        Returns
        -------
        ZCategoricalCategories
            Deserialized categorical categories dtype instance

        Raises
        ------
        ValueError
            If JSON format is invalid or inner_dtype unsupported
        """
        if not cls._check_json_v3(data):
            msg = f"Invalid v3 JSON for {cls._zarr_v3_name}: {data}"
            raise DataTypeValidationError(msg)

        # Extract and validate inner_dtype
        config = data["configuration"]  # type: ignore[index]
        inner_dtype = config["inner_dtype"]

        if inner_dtype not in ("int64", "float64", "string"):
            msg = f"Invalid inner_dtype '{inner_dtype}', must be one of: int64, float64, string"
            raise DataTypeValidationError(msg)

        return cls(inner_dtype=inner_dtype)  # type: ignore[arg-type]

    @classmethod
    def _check_json_v2(cls, _data: DTypeJSON) -> TypeGuard[dict]:
        """Zarr v2 not supported - always returns False.

        Parameters
        ----------
        _data : str or dict
            JSON data to validate (unused)

        Returns
        -------
        bool
            Always False (v2 not supported)
        """
        return False

    @classmethod
    def _from_json_v2(cls, _data: DTypeJSON) -> ZNarwhalsCategoricalCategories:
        """Zarr v2 not supported - always raises error.

        Parameters
        ----------
        data : dict
            JSON dict

        Raises
        ------
        DataTypeValidationError
            Always raised - v2 not supported
        """
        msg = "ZCategoricalCategories only supports Zarr v3, not v2"
        raise DataTypeValidationError(msg)

    @classmethod
    def from_native_dtype(cls, dtype: np.dtype) -> ZNarwhalsCategoricalCategories:
        """Convert NumPy dtype to ZCategoricalCategories.

        This method should NOT be used for automatic dtype inference from regular arrays.
        It will raise an error to prevent conflicts with standard Int64/Float64/String dtypes.

        ZCategoricalCategories should only be created explicitly via the constructor
        with the desired inner_dtype specification.

        Parameters
        ----------
        dtype : np.dtype
            NumPy dtype of categories array

        Returns
        -------
        ZCategoricalCategories
            Never - always raises error

        Raises
        ------
        DataTypeValidationError
            Always raised - this dtype requires explicit configuration via constructor
        """
        msg = f"ZCategoricalCategories cannot be inferred from numpy dtype {dtype}. "
        raise DataTypeValidationError(msg)

    def to_native_dtype(self) -> np.dtype:
        """Convert to NumPy dtype based on inner_dtype.

        Returns
        -------
        np.dtype
            int64, float64, or StringDType based on configuration
        """
        if self.inner_dtype == "int64":
            return np.dtype(np.int64)
        elif self.inner_dtype == "float64":
            return np.dtype(np.float64)
        else:
            from numpy.dtypes import StringDType

            return np.dtype(StringDType())

    def _check_scalar(self, data: object) -> TypeGuard[int | float | str]:
        """Check if data is valid scalar for this category dtype.

        Parameters
        ----------
        data : object
            Data to check

        Returns
        -------
        bool
            True if data matches inner_dtype
        """
        if self.inner_dtype == "int64":
            return isinstance(data, (int, np.integer))
        elif self.inner_dtype == "float64":
            return isinstance(data, (float, np.floating))
        else:
            return isinstance(data, str)

    def _cast_scalar_unchecked(self, data: int | float | str) -> np.int64 | np.float64 | str:
        """Cast scalar without validation.

        Parameters
        ----------
        data : int, float, or str
            Scalar value to cast

        Returns
        -------
        np.int64, np.float64, or str
            Casted scalar matching inner_dtype
        """
        if self.inner_dtype == "int64":
            return np.int64(data)
        elif self.inner_dtype == "float64":
            return np.float64(data)
        else:
            return str(data)

    def cast_scalar(self, data: object) -> np.int64 | np.float64 | str:
        """Cast object to category scalar.

        Parameters
        ----------
        data : object
            Object to cast

        Returns
        -------
        np.int64, np.float64, or str
            Category value matching inner_dtype

        Raises
        ------
        TypeError
            If data cannot be cast to inner_dtype
        """
        if not self._check_scalar(data):
            msg = f"Cannot cast {type(data)} to category type {self.inner_dtype}"
            raise TypeError(msg)
        return self._cast_scalar_unchecked(data)

    def default_scalar(self) -> np.int64 | np.float64 | str:
        """Get default scalar value based on inner_dtype.

        Returns
        -------
        np.int64, np.float64, or str
            Default value: 0 for int64, 0.0 for float64, "" for string
        """
        if self.inner_dtype == "int64":
            return np.int64(0)
        elif self.inner_dtype == "float64":
            return np.float64(0.0)
        else:
            return ""

    def from_json_scalar(self, data: JSON, *, zarr_format: ZarrFormat) -> np.int64 | np.float64 | str:
        """Deserialize scalar from JSON.

        Parameters
        ----------
        data : JSON
            JSON data
        zarr_format : ZarrFormat
            Zarr format version

        Returns
        -------
        np.int64, np.float64, or str
            Deserialized scalar
        """
        return self.cast_scalar(data)

    def to_json_scalar(self, data: object, *, zarr_format: ZarrFormat) -> JSON:
        """Serialize scalar to JSON.

        Parameters
        ----------
        data : object
            Scalar value
        zarr_format : ZarrFormat
            Zarr format version

        Returns
        -------
        JSON
            JSON-serializable value (int, float, or str)
        """
        casted = self.cast_scalar(data)
        # Return as native Python type for JSON
        if self.inner_dtype == "int64":
            return int(casted)
        elif self.inner_dtype == "float64":
            return float(casted)
        else:
            return str(casted)


def get_categorical_codes(series) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract integer codes from categorical series.

    Pure Narwhals + NumPy implementation without pandas/polars imports.
    Replicates pandas .cat.codes behavior for backend-agnostic categorical encoding.

    Parameters
    ----------
    series
        Narwhals Series with Categorical or Enum dtype

    Returns
    -------
    tuple[np.ndarray, np.ndarray, bool]
        Tuple of (codes, categories, ordered):
        - codes: int32 array where valid values map to 0..len(categories)-1,
          missing/NaN values map to -1
        - categories: numpy array of unique category values
        - ordered: bool indicating if categorical is ordered

    Examples
    --------
    >>> import pandas as pd
    >>> import narwhals as nw
    >>> cat = pd.Categorical(["A", "B", "A"], categories=["A", "B", "C"])
    >>> s = nw.from_native(pd.Series(cat), series_only=True)
    >>> codes, cats, ordered = get_categorical_codes(s)
    >>> codes
    array([0, 1, 0], dtype=int32)
    >>> cats
    array(['A', 'B', 'C'], dtype=object)
    >>> ordered
    False
    """
    import narwhals as nw

    categories = series.cat.get_categories().to_numpy()
    values = series.to_numpy()
    ordered = nw.is_ordered_categorical(series)

    # Create hash map for O(1) lookup: category -> code
    cat_to_code = {cat: idx for idx, cat in enumerate(categories)}

    # Allocate result array
    codes = np.empty(len(values), dtype=np.int32)

    # Map values to codes
    for i, value in enumerate(values):
        # Handle None or NaN (missing values become -1)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            codes[i] = -1
        else:
            codes[i] = cat_to_code.get(value, -1)

    return codes, categories, ordered
