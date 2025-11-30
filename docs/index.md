---
title: zarrwhals
summary: DataFrame-agnostic Zarr storage powered by Narwhals
description: Store pandas, Polars, and Dask DataFrames in Zarr format with automatic type preservation and lazy loading.
keywords: zarr, dataframe, pandas, polars, dask, narwhals, storage
order: 1
---

# zarrwhals

Bring your DataFrame to Zarr.

Currently works with:

- Pandas DataFrames
- Polars (DataFrames & LazyFrames)
- Dask DataFrames

---

## Quick Start

/// tab | pandas
    new: true

    :::python
    import pandas as pd
    import zarrwhals as zw

    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, 30.1],
        "category": pd.Categorical(["A", "B", "A"])
    })

    zw.to_zarr(df, "data.zarr")
    df_loaded = zw.from_zarr("data.zarr")
///

/// tab | polars

    :::python
    import polars as pl
    import zarrwhals as zw

    df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "value": [10.5, 20.3, 30.1]
    })

    zw.to_zarr(df, "data.zarr")
    df_loaded = zw.from_zarr("data.zarr", backend="polars")
///

---

## Installation

/// tab | pip
    new: true

    :::bash
    pip install git+https://github.com/srivarra/zarrwhals.git@main
///

/// tab | uv

    :::bash
    uv add git+https://github.com/srivarra/zarrwhals.git@main
///

---


## How It Works

```text
pandas / polars / dask  →  Narwhals  →  zarrwhals  →  Zarr Storage
```

zarrwhals serializes DataFrames via [Narwhals](https://narwhals-dev.github.io/narwhals/) into Zarr stores, preserving type information in an interchange format. This allows reading back as pandas, Polars, or Dask—regardless of what library wrote the data.

---

## Next Steps

- [API Reference](api.md) — Function documentation
- [Architecture](architecture.md) — How zarrwhals works
- [Contributing](contributing.md) — Get involved
