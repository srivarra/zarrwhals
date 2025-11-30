"""Zarrwhals: DataFrame-agnostic Zarr storage powered by Narwhals.

Simple API for storing DataFrames in Zarr format with efficient chunking and compression.
"""

from importlib.metadata import version

# Register custom dtypes with Zarr before any operations
from . import zdtypes as _zdtypes
from .io import from_zarr, get_spec, to_zarr

__all__ = [
    "from_zarr",
    "get_spec",
    "to_zarr",
]

__version__ = version("zarrwhals")

import zarr
import zarrs

zarr.config.set(
    {
        "threading.max_workers": None,
        "array.write_empty_chunks": False,
        "codec_pipeline": {
            "path": "zarrs.ZarrsCodecPipeline",
            "batch_size": 1,
            "validate_checksums": True,
            "store_empty_chunks": False,
            "chunk_concurrent_maximum": None,
            "chunk_concurrent_minimum": 4,
        },
    }
)
