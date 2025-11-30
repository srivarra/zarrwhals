---
title: Architecture
summary: How zarrwhals works under the hood
description: Learn about zarrwhals internals, data flow, storage format, and type system.
keywords: architecture, zarr, narwhals, storage format, type system, lazy loading
order: 3
---

# Architecture

So, how does this work?

## Overview

**zarrwhals** bridges DataFrame libraries with Zarr storage. It handles serialization and deserialization while delegating computation to your preferred DataFrame engine.

## Narwhals Integration

zarrwhals uses [Narwhals](https://narwhals-dev.github.io/narwhals/) for DataFrame-agnostic operations, specifically it converts your DataFrame to a [Narwhals DataFrame][nw-df-docs],
then this intermediate representation gets written out to Zarr.

```text
pandas / polars / dask  →  Narwhals  →  zarrwhals  →  Zarr Storage
```

## Data Flow

### Writing DataFrames

1. Call `zw.to_zarr(df, "store.zarr")`
2. Convert to Narwhals DataFrame
3. For each column: encode data + metadata → create Zarr array (or group for categoricals)
4. Write group metadata
5. Zarr Store created

### Reading DataFrames

1. Call `zw.from_zarr("store.zarr", backend="polars")`
2. Create an Internal [Narwhals complient][nw-extensions] `ZarrFrame`, then convert to the user's specified `backend`.
3. Return DataFrame

## Storage Format

`zarrwhals` follows the [anndata on-disk specification](https://anndata.readthedocs.io/en/stable/fileformat-prose.html#dataframes) for DataFrame storage, enabling interoperability with the scientific Python ecosystem.

DataFrames are stored as a Zarr group:

```text
store.zarr/
├── zarr.json              # Group metadata
├── _index/                # Row index array
├── column_a/              # Regular column (array)
│   └── zarr.json          # Array metadata + narwhals_dtype
├── column_b/              # Categorical column (group)
│   ├── codes/             # Integer codes array
│   └── categories/        # Category labels array
└── ...
```

### Group Attributes

```json
{
    "encoding-type": "dataframe",
    "encoding-version": "0.0.1",
    "column-order": ["column_a", "column_b"],
    "_index": "_index"
}
```

### Column Attributes

Each column stores its Narwhals dtype for accurate round-trip:

```json
{
    "encoding-type": "array",
    "narwhals_dtype": "Int64"
}
```

## Type System

zarrwhals implements custom Zarr dtypes for types not natively supported:

| Source Type | Zarr Encoding |
|-------------|---------------|
| int8/16/32/64 | Native |
| float32/64 | Native |
| bool | Native |
| String | VLen UTF-8 |
| Categorical | Codes + Categories arrays |
| Datetime | int64 + unit metadata |
| Duration | int64 + unit metadata |
| Object | Pickle bytes |

### Categorical Encoding

Categoricals use [code/category separation](https://anndata.readthedocs.io/en/stable/fileformat-prose.html#categorical-arrays):

One Zarr Array gets written for the codes, another for the categories. During read time,
the codes get mapped to the original categories reconstructing the categorical column.

```text
Input: ['A', 'B', 'A', 'C', 'B']
  ↓
codes:      [0, 1, 0, 2, 1]      (int array)
categories: ['A', 'B', 'C']      (string array)
ordered:    false
```

## General Zarr Features in `zarrwhals`

### Chunking

Control how data is split across storage:

```python
# Default: Zarr auto-selects chunk size
zw.to_zarr(df, "store.zarr")

# Custom: 10,000 rows per chunk
zw.to_zarr(df, "store.zarr", chunks=10_000)
```

### Sharding

Group multiple chunks into single files for large datasets:

```python
# 1,000 rows per chunk, 100,000 rows per shard
zw.to_zarr(df, "store.zarr", chunks=1_000, shards=100_000)
```

### Compression

Choose from multiple codecs:

```python
from zarr.codecs import ZstdCodec, LZ4Codec, GzipCodec

# Auto (Zarr default)
zw.to_zarr(df, "store.zarr", compressors="auto")

# Custom codec with options
zw.to_zarr(df, "store.zarr", compressors=ZstdCodec(level=5))

# No compression
zw.to_zarr(df, "store.zarr", compressors=None)
```

## Next Steps

- See the [API Reference](api.md) for read/write function documentation.
- Check out [Contributing](contributing.md) to get involved

[nw-df-docs]: https://narwhals-dev.github.io/narwhals/basics/dataframe/#__tabbed_2_1
[nw-extensions]: https://narwhals-dev.github.io/narwhals/extending/
