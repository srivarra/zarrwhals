"""Lazy DataFrame backed by Zarr storage.

Provides ZarrFrame, an internal lazy DataFrame implementation that defers
I/O until materialization. This is used internally by from_zarr() to optimize
column projection and lazy loading.

Uses Zarr's async API for parallel column reads when collecting data.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import narwhals as nw
import numpy as np
import zarr
from zarr.storage import StoreLike

import zarrwhals
from zarrwhals._stores import resolve_store
from zarrwhals.specs import DataFrameGroupSpec

if TYPE_CHECKING:
    from typing import Self

    import dask.array as da
    import dask.dataframe as dd
    import pandas as pd
    import polars as pl


class ZarrFrame:
    """Internal DataFrame backed by Zarr storage.

    Defers I/O operations until materialization via `collect()`.
    It is used internally by `from_zarr()` and should not be instantiated directly.

    Parameters
    ----------
    store
        Zarr store path or StoreLike object.

    Notes
    -----
    - Construction only reads Zarr metadata (no array data)
    - Column selection via `select()` is lazy (no I/O)
    - Row slicing via `__getitem__` is lazy (no I/O)
    - Data is only read when `collect()`, `to_dask()`, or `to_polars_lazy()` is called
    """

    def __init__(self, store: StoreLike) -> None:
        self._store, _ = resolve_store(store, require_exists=True)
        self._root = zarr.open_group(self._store, mode="r")
        self._spec = DataFrameGroupSpec.from_zarr(self._root)

        # Lazy state - no data loaded yet
        self._selected_columns: list[str] | None = None  # None = all columns
        self._row_slice: slice | None = None

    def _copy(self) -> ZarrFrame:
        """Create a shallow copy for lazy operations.

        Returns
        -------
        ZarrFrame
            New instance with same metadata but independent lazy state.
        """
        new = ZarrFrame.__new__(ZarrFrame)
        new._store = self._store
        new._root = self._root
        new._spec = self._spec
        new._selected_columns = self._selected_columns
        new._row_slice = self._row_slice
        return new

    def __narwhals_lazyframe__(self) -> Self:
        """Narwhals protocol - identifies this as a lazy frame.

        Returns
        -------
        Self
            Returns self to indicate this is a lazy frame.
        """
        return self

    def __native_namespace__(self) -> ModuleType:
        """Return the zarrwhals module for backend identification.

        Returns
        -------
        ModuleType
            The zarrwhals module.
        """
        return zarrwhals

    def select(self, columns: Sequence[str]) -> ZarrFrame:
        """Select columns (lazy operation - no I/O).

        Parameters
        ----------
        columns
            Column names to select.

        Returns
        -------
        ZarrFrame
            New ZarrFrame with column selection applied.

        Raises
        ------
        ValueError
            If any requested column doesn't exist.
        """
        all_columns_set = set(self._spec.attributes.column_order)
        invalid_cols = [c for c in columns if c not in all_columns_set]
        if invalid_cols:
            msg = f"Requested columns not found: {invalid_cols}. Available: {list(all_columns_set)}"
            raise ValueError(msg)

        new = self._copy()
        new._selected_columns = list(columns)
        return new

    def __getitem__(self, key: slice) -> ZarrFrame:
        """Row slicing.

        Parameters
        ----------
        key
            Slice object for row selection.

        Returns
        -------
        ZarrFrame
            New ZarrFrame with row slice applied.

        Raises
        ------
        TypeError
            If key is not a slice.
        """
        if not isinstance(key, slice):
            msg = "Only slice indexing is supported"
            raise TypeError(msg)

        new = self._copy()
        new._row_slice = key
        return new

    @property
    def columns(self) -> list[str]:
        """Available columns (from metadata, no I/O).

        Returns
        -------
        list[str]
            Column names in the frame.
        """
        if self._selected_columns is not None:
            return self._selected_columns
        return self._spec.attributes.column_order

    @property
    def schema(self) -> dict[str, type]:
        """Column dtypes from Zarr metadata (no I/O).

        Returns
        -------
        dict[str, type]
            Mapping of column names to Narwhals dtype classes.
        """
        from zarrwhals.zdtypes.converters import zdtype_to_narwhals

        schema: dict[str, type] = {}
        for col in self.columns:
            elem = self._root[col]
            if isinstance(elem, zarr.Group):
                # Categorical or Enum group
                encoding_type = elem.attrs.get("encoding-type", "categorical")
                if encoding_type == "enum":
                    schema[col] = nw.Enum
                else:
                    schema[col] = nw.Categorical
            elif hasattr(elem, "metadata") and hasattr(elem.metadata, "dtype"):
                nw_dtype = zdtype_to_narwhals(elem.metadata.dtype)
                schema[col] = type(nw_dtype) if nw_dtype else nw.Unknown
            else:
                schema[col] = nw.Unknown
        return schema

    def collect(
        self,
        backend: Literal["pandas", "polars"] = "polars",
    ) -> pd.DataFrame | pl.DataFrame:
        """Load data from Zarr and return concrete DataFrame.

        This is where I/O happens - data is read from Zarr storage.
        Only requested columns and row slices are read. Uses async parallel
        reads for improved performance with many columns.

        Parameters
        ----------
        backend
            Target backend: "pandas" or "polars" (default: "polars").

        Returns
        -------
        pd.DataFrame | pl.DataFrame
            Materialized DataFrame in the requested backend.
        """
        from zarrwhals.zdtypes.converters import zdtype_to_narwhals

        data_dict, dtype_metadata = asyncio.run(self._collect_columns_async())

        nw_df = nw.from_dict(data_dict, backend=backend)

        for col_name, (dtype_type, dtype_info) in dtype_metadata.items():
            try:
                if dtype_type == "enum":
                    nw_dtype = nw.Enum(dtype_info)
                elif dtype_type == "categorical":
                    nw_dtype = nw.Categorical
                elif dtype_type == "zdtype":
                    nw_dtype = zdtype_to_narwhals(dtype_info)
                    if nw_dtype is None:
                        continue
                else:
                    continue
                nw_df = nw_df.with_columns(nw.col(col_name).cast(nw_dtype))
            except (TypeError, ValueError, AttributeError) as e:
                warnings.warn(
                    f"Failed to cast column '{col_name}' to {dtype_type}: {e}. Keeping original dtype.",
                    stacklevel=2,
                )

        # Return native DataFrame
        if backend == "pandas":
            return nw_df.to_pandas()
        return nw_df.to_polars()

    async def _collect_columns_async(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, object]]]:
        """Read all columns in parallel using Zarr's async API.

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, tuple[str, object]]]
            Tuple of (data_dict, dtype_metadata) for DataFrame construction.
            Column order is preserved to match self.columns.
        """
        cols = self.columns

        # Separate regular columns from categorical groups
        regular_cols = []
        categorical_cols = []

        for col_name in cols:
            elem = self._root[col_name]
            if isinstance(elem, zarr.Group):
                categorical_cols.append(col_name)
            else:
                regular_cols.append(col_name)

        # Read regular columns in parallel using async
        tasks = [self._read_column_async(col_name) for col_name in regular_cols]
        results = await asyncio.gather(*tasks)

        # Build intermediate dicts from results
        results_dict: dict[str, np.ndarray] = {}
        dtype_metadata: dict[str, tuple[str, object]] = {}

        for col_name, col_data, dtype_info in results:
            results_dict[col_name] = col_data
            if dtype_info is not None:
                dtype_metadata[col_name] = dtype_info

        # Read categorical columns (sync - they use groups not arrays)
        for col_name in categorical_cols:
            elem = self._root[col_name]
            col_data = self._decode_categorical(elem)
            ordered = elem.attrs.get("ordered", False)
            if ordered:
                dtype_metadata[col_name] = ("enum", list(elem["categories"][...]))
            else:
                dtype_metadata[col_name] = ("categorical", None)
            results_dict[col_name] = col_data

        # Rebuild data_dict in original column order
        data_dict: dict[str, np.ndarray] = {col: results_dict[col] for col in cols}

        return data_dict, dtype_metadata

    async def _read_column_async(self, col_name: str) -> tuple[str, np.ndarray, tuple[str, object] | None]:
        """Read a single column asynchronously.

        Parameters
        ----------
        col_name
            Name of the column to read.

        Returns
        -------
        tuple[str, np.ndarray, tuple[str, object] | None]
            Tuple of (column_name, data, dtype_metadata).
        """
        elem = self._root[col_name]
        async_arr = elem.async_array

        # Read with row slice using async API
        if self._row_slice is not None:
            col_data = await async_arr.getitem(self._row_slice)
        else:
            col_data = await async_arr.getitem(slice(None))

        # Apply dtype transformation
        dtype_info = None
        dtype_str = elem.attrs.get("narwhals_dtype")
        if dtype_str and hasattr(elem.metadata, "dtype"):
            col_data = self._decode_dtype(col_data, elem.metadata.dtype)
            dtype_info = ("zdtype", elem.metadata.dtype)

        return col_name, col_data, dtype_info

    def _decode_categorical(self, group: zarr.Group) -> np.ndarray:
        """Decode categorical/enum group to numpy array.

        Parameters
        ----------
        group
            Zarr group with 'codes' and 'categories' arrays.

        Returns
        -------
        np.ndarray
            Object array with category values (None for missing).
        """
        # Apply row slice to codes for lazy loading
        if self._row_slice is not None:
            codes = group["codes"][self._row_slice]
        else:
            codes = group["codes"][...]

        # Categories are small, always read fully
        categories = group["categories"][...]

        result = np.empty(len(codes), dtype=object)
        valid_mask = codes >= 0
        result[valid_mask] = categories[codes[valid_mask]]
        result[np.logical_not(valid_mask)] = None

        return result

    def _decode_dtype(self, data: np.ndarray, zarr_dtype: object) -> np.ndarray:
        """Apply dtype transformation to data array.

        Parameters
        ----------
        data
            Raw numpy array from Zarr.
        zarr_dtype
            ZDType metadata from Zarr array.

        Returns
        -------
        np.ndarray
            Transformed array with correct dtype.
        """
        from zarrwhals.zdtypes import (
            ZNarwhalsDate,
            ZNarwhalsDatetime,
            ZNarwhalsDuration,
            ZNarwhalsTime,
        )

        # ZNarwhalsDatetime - convert int64 to datetime64
        if isinstance(zarr_dtype, ZNarwhalsDatetime):
            unit = zarr_dtype.time_unit
            return data.view(f"datetime64[{unit}]")

        # ZNarwhalsDuration - convert int64 to timedelta64
        if isinstance(zarr_dtype, ZNarwhalsDuration):
            unit = zarr_dtype.time_unit
            return data.view(f"timedelta64[{unit}]")

        # ZNarwhalsDate - convert int32 to datetime64[D]
        if isinstance(zarr_dtype, ZNarwhalsDate):
            return data.view("datetime64[D]")

        # ZNarwhalsTime - keep as int64 (nanoseconds since midnight)
        if isinstance(zarr_dtype, ZNarwhalsTime):
            return data

        # All other types: return as-is
        return data

    def _decode_categorical_to_dask(self, categorical_group: zarr.Group) -> da.Array:
        """Decode categorical Zarr group to Dask array lazily.

        Parameters
        ----------
        categorical_group
            Zarr group containing 'codes' and 'categories' arrays.

        Returns
        -------
        da.Array
            Dask array with decoded category values (object dtype).
        """
        import dask.array as da

        codes_array = categorical_group["codes"]
        categories_array = categorical_group["categories"]

        # Load categories eagerly (small)
        categories = categories_array[...]

        # Create lazy Dask array from codes
        codes_da = da.from_zarr(codes_array)

        # Map codes to categories lazily using map_blocks
        def decode_codes_block(codes_chunk, categories=categories):
            """Decode a chunk of codes to category values."""
            result = np.empty(len(codes_chunk), dtype=object)
            valid_mask = codes_chunk >= 0
            result[valid_mask] = categories[codes_chunk[valid_mask]]
            result[np.logical_not(valid_mask)] = None
            return result

        # Apply decoding lazily across all chunks
        return da.map_blocks(
            decode_codes_block,
            codes_da,
            dtype=object,
            drop_axis=[],
        )

    def to_dask(self) -> dd.DataFrame:
        """Return Dask DataFrame backed by Zarr arrays (lazy).

        Uses dask.array.from_zarr() which maps Zarr chunks to Dask partitions.

        Returns
        -------
        dd.DataFrame
            Dask DataFrame with lazy loading.
        """
        import dask.array as da
        import dask.dataframe as dd

        dask_series = []
        for col in self.columns:
            zarr_elem = self._root[col]
            if isinstance(zarr_elem, zarr.Group):
                # Categorical column: decode codes to categories lazily
                decoded_da = self._decode_categorical_to_dask(zarr_elem)
                # Convert to Dask DataFrame Series
                series_df = dd.from_dask_array(decoded_da, columns=[col])
            else:
                # Regular column: direct conversion
                series_df = dd.from_dask_array(da.from_zarr(zarr_elem), columns=[col])
            dask_series.append(series_df)

        return dd.concat(dask_series, axis=1)

    def to_polars_lazy(self) -> pl.LazyFrame:
        """Return Polars LazyFrame backed by our loader.

        Operations on the LazyFrame will be deferred until `.collect()`.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame that materializes data on collect.
        """
        import polars as pl

        return pl.LazyFrame(self.collect("polars")).lazy()
