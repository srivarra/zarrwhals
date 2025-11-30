"""DataFrame encoding to Zarr storage.

Encodes DataFrames to Zarr using column-wise storage pattern.
Supports pandas and polars DataFrames via Narwhals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
import numpy as np
import zarr
from zarr.core.array import CompressorsLike

from ._stores import resolve_store
from .specs import create_dataframe_metadata, validate_column_names

if TYPE_CHECKING:
    from zarr.storage import StoreLike

#: Write mode type for Zarr stores
ZarrWriteMode = Literal["w-", "w"]


def encode_dataframe(
    df: Any,
    store: StoreLike,
    *,
    chunks: int | Literal["auto"] | None = "auto",
    shards: int | None = None,
    compressors: CompressorsLike = "auto",
    mode: ZarrWriteMode = "w-",
) -> zarr.Group:
    """Encode DataFrame to Zarr storage.

    Parameters
    ----------
    df
        DataFrame (pandas, polars, or Narwhals).
    store
        Path or store object.
    chunks
        Chunk size in rows, or "auto" (default: "auto").
    shards
        Shard size in rows (default: None).
    compressors
        Compressor codec(s) (default: "auto").
    mode
        Write mode (default: "w-"):

        - "w-": Create new store, fail if exists (safe default)
        - "w": Overwrite existing store completely

    Returns
    -------
    zarr.Group
        Created Zarr group.

    Raises
    ------
    FileExistsError
        If mode="w-" and store already exists.
    TypeError
        If DataFrame type unsupported.
    """
    try:
        nw_df = nw.from_native(df, eager_only=True)
    except Exception as e:
        msg = f"Failed to convert DataFrame to Narwhals: {e}"
        raise TypeError(msg) from e

    column_names = nw_df.columns
    validate_column_names(column_names)

    index_name = "_index"

    # For paths, check existence BEFORE creating LocalStore (for mode="w-")
    from os import PathLike
    from pathlib import Path

    if mode == "w-" and isinstance(store, (str, PathLike)):
        path = Path(store)
        if path.exists():
            msg = f"Store already exists at {store}. Use mode='w' to overwrite."
            raise FileExistsError(msg)

    resolved_store, original_path = resolve_store(store, mkdir=True)

    if mode == "w-" and original_path is None:
        # Non-path stores: try to open to detect existence
        try:
            zarr.open_group(store=resolved_store, mode="r")
            msg = "Store already contains a group. Use mode='w' to overwrite."
            raise FileExistsError(msg)
        except (zarr.errors.GroupNotFoundError, KeyError):
            pass  # Doesn't exist, proceed

    # Create or overwrite the group
    try:
        root = zarr.open_group(store=resolved_store, mode="w", zarr_format=3)
    except Exception as e:
        msg = f"Failed to create group in store: {e}"
        raise ValueError(msg) from e

    metadata = create_dataframe_metadata(column_names, index_name)
    for key, value in metadata.model_dump(by_alias=True).items():
        root.attrs[key] = value

    index_data = np.arange(len(nw_df))
    backend_impl = nw_df.implementation
    backend = backend_impl.to_native_namespace().__name__
    index_series = nw.new_series(index_name, index_data, backend=backend)

    df_with_index = nw_df.with_columns(index_series)
    all_columns = [index_name, *column_names]

    from .writers import encode_series_narwhals

    for col_name in all_columns:
        encode_series_narwhals(
            root, col_name, df_with_index[col_name], chunks=chunks, shards=shards, compressors=compressors
        )

    return root
