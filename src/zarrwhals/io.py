"""Main API for Zarr DataFrame storage.

Provides simple `to_zarr()` and `from_zarr()` functions for DataFrame persistence.
Works with any DataFrame backend supported by Narwhals (pandas, polars, etc.).
"""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal, overload

from zarr.core.array import CompressorsLike
from zarr.storage import StoreLike

from ._stores import resolve_store
from .encoders import ZarrWriteMode, encode_dataframe
from .specs import DataFrameGroupSpec, validate_dataframe_store
from .zarrframe import ZarrFrame

if TYPE_CHECKING:
    import dask.dataframe as dd
    import pandas as pd
    import polars as pl
    from narwhals.typing import IntoDataFrame


def to_zarr(
    df: IntoDataFrame,
    store: PathLike[str] | StoreLike,
    *,
    chunks: int | Literal["auto"] | None = "auto",
    shards: int | None = None,
    compressors: CompressorsLike = "auto",
    mode: ZarrWriteMode = "w-",
) -> None:
    """Write a DataFrame to Zarr storage.

    Parameters
    ----------
    df
        DataFrame to write (pandas, polars, anything that can be converted to a Narwhals DataFrame).
    store
        Path or store object.
    chunks
        Chunk size in rows, or "auto" to let Zarr decide (default: "auto").
    shards
        Shard size in rows (default: None).
    compressors
        Compressor codec(s). Can be "auto" (Zarr default), a Zarr codec object,
        or None for no compression (default: "auto").
    mode
        Write mode (default: "w-"):

        - "w-": Create new store, fail if exists (safe default)
        - "w": Overwrite existing store completely

    Raises
    ------
    FileExistsError
        If mode="w-" and store already exists.
    TypeError
        If DataFrame type not supported.

    Notes
    -----
    For custom compression, pass a Zarr codec object (e.g., ``ZstdCodec(level=5)``).
    See the `Zarr compressors documentation
    <https://zarr.readthedocs.io/en/stable/user-guide/arrays/#compressors>`_
    for available codecs and configuration options.

    Examples
    --------
    Create new store (fails if exists):

    >>> import pandas as pd
    >>> import zarrwhals as zw
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    >>> zw.to_zarr(df, "data.zarr")  # mode="w-" default

    Overwrite existing store:

    >>> zw.to_zarr(df, "data.zarr", mode="w")

    With chunking and sharding:

    >>> zw.to_zarr(df, "data.zarr", chunks=5000, shards=50000, mode="w")
    """
    encode_dataframe(
        df,
        store,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        mode=mode,
    )


# Pandas: always eager, no lazy param exposed
@overload
def from_zarr(
    store: PathLike[str] | StoreLike,
    *,
    backend: Literal["pandas"],
    columns: list[str] | None = None,
) -> pd.DataFrame: ...


# Polars: lazy=True (explicit) → LazyFrame
@overload
def from_zarr(
    store: PathLike[str] | StoreLike,
    *,
    backend: Literal["polars"],
    columns: list[str] | None = None,
    lazy: Literal[True] = True,
) -> pl.LazyFrame: ...


@overload
def from_zarr(
    store: PathLike[str] | StoreLike,
    *,
    backend: Literal["polars"],
    columns: list[str] | None = None,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


# Dask: always lazy, no lazy param exposed
@overload
def from_zarr(
    store: PathLike[str] | StoreLike,
    *,
    backend: Literal["dask"],
    columns: list[str] | None = None,
) -> dd.DataFrame: ...


def from_zarr(
    store: PathLike[str] | StoreLike,
    *,
    backend: Literal["pandas", "polars", "dask"] = "pandas",
    columns: list[str] | None = None,
    lazy: bool | None = None,
) -> pd.DataFrame | pl.DataFrame | pl.LazyFrame | dd.DataFrame:
    """Read DataFrame from Zarr storage.

    Parameters
    ----------
    store
        Path or store object.
    backend
        Target backend: "pandas", "polars", or "dask" (default: "pandas").
    columns
        Optional columns to read (default: all).
    lazy
        For polars only: return LazyFrame (True) or DataFrame (False).
        Defaults to True for polars. Not applicable to pandas or dask.

    Returns
    -------
    pd.DataFrame | pl.DataFrame | pl.LazyFrame | dd.DataFrame
        DataFrame or LazyFrame in requested backend.

    Raises
    ------
    ValueError
        If store missing, columns not found, or invalid lazy/backend combination.
    FileNotFoundError
        If store path doesn't exist.

    Examples
    --------
    >>> import zarrwhals as zw
    >>> df = zw.from_zarr("data.zarr", backend="pandas")
    >>> lf = zw.from_zarr("data.zarr", backend="polars")
    >>> df = zw.from_zarr("data.zarr", backend="polars", lazy=False)
    >>> ddf = zw.from_zarr("data.zarr", backend="dask")
    >>> df = zw.from_zarr("data.zarr", backend="pandas", columns=["a", "c"])
    """
    # Validate backend/lazy combinations
    if backend == "pandas" and lazy is True:
        msg = "pandas backend does not support lazy=True"
        raise ValueError(msg)
    if backend == "dask" and lazy is False:
        msg = "dask backend does not support lazy=False (dask is always lazy)"
        raise ValueError(msg)

    # Set backend-specific defaults for lazy parameter
    if lazy is None:
        lazy = backend in ("polars", "dask")

    # Create internal lazy frame (no I/O yet - only reads metadata)
    lzf = ZarrFrame(store)

    # Apply column selection if specified (still no I/O)
    if columns is not None:
        lzf = lzf.select(columns)

    # Materialize based on backend + lazy flag
    match (backend, lazy):
        case ("dask", _):
            return lzf.to_dask()
        case ("polars", True):
            return lzf.to_polars_lazy()
        case ("polars", False):
            return lzf.collect("polars")
        case ("pandas", _):
            return lzf.collect("pandas")
        case _:
            msg = f"Unsupported backend/lazy combination: {backend}, {lazy}"
            raise ValueError(msg)


def get_spec(store: PathLike[str] | StoreLike) -> DataFrameGroupSpec:
    """Get DataFrameGroupSpec from a Zarr store.

    Parameters
    ----------
    store
        Path or store object.

    Returns
    -------
    DataFrameGroupSpec
        Validated spec with metadata and structure.

    Raises
    ------
    ValueError
        If store structure invalid.
    FileNotFoundError
        If store path doesn't exist.

    Examples
    --------
    >>> import zarrwhals as zw
    >>> spec = zw.get_spec("data.zarr")
    >>> print(spec.attributes.column_order)
    ['a', 'b', 'c']
    """
    import zarr

    # Resolve store - converts paths to ObjectStore(LocalStore(...))
    # require_exists=True ensures FileNotFoundError before LocalStore canonicalization
    resolved_store, _original_path = resolve_store(store, require_exists=True)

    return validate_dataframe_store(zarr.open_group(store=resolved_store, mode="r"))


__all__ = ["from_zarr", "get_spec", "to_zarr"]
