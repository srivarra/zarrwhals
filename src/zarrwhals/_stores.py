"""Store utilities for zarrwhals."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from obstore.store import LocalStore
from zarr.storage import ObjectStore

if TYPE_CHECKING:
    from zarr.storage import StoreLike


def resolve_store(
    store: PathLike[str] | StoreLike,
    *,
    mkdir: bool = False,
    require_exists: bool = False,
) -> tuple[StoreLike, Path | None]:
    """Resolve store to a StoreLike object.

    Converts path-based stores to ObjectStore(LocalStore(...)) internally
    for unified store handling.

    Parameters
    ----------
    store
        Path or store object.
    mkdir
        If True and store is a path, create the directory.
    require_exists
        If True and store is a path, raise FileNotFoundError if path doesn't exist.
        Must check BEFORE LocalStore creation to avoid canonicalization errors.

    Returns
    -------
    tuple[StoreLike, Path | None]
        Resolved store and original path (if path was provided).

    Raises
    ------
    FileNotFoundError
        If require_exists=True and path doesn't exist.
    """
    if isinstance(store, (str, PathLike)):
        path = Path(store)
        if require_exists and not path.exists():
            msg = f"Zarr store not found at {path}"
            raise FileNotFoundError(msg)
        local_store = LocalStore(prefix=str(path.resolve()), mkdir=mkdir)
        return ObjectStore(local_store), path
    else:
        return store, None
