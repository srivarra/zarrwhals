# zarrwhals

| | |
|---:|:---|
| **Package** | [![PyPI][pypi-badge]][pypi-url] [![Python][python-badge]][pypi-url] [![License][license-badge]][license-url] |
| **CI/CD** | [![Tests][tests-badge]][tests-url] [![Build][build-badge]][build-url] [![codecov][codecov-badge]][codecov-url] [![CodSpeed][codspeed-badge]][codspeed-url] |
| **Docs** | [![Docs][docs-badge]][docs-url] |
| **Tools** | [![Ruff][ruff-badge]][ruff-url] [![uv][uv-badge]][uv-url] [![gitmoji][gitmoji-badge]][gitmoji-url] |

<!-- Badges -->
[pypi-badge]: https://img.shields.io/pypi/v/zarrwhals?logo=pypi&labelColor=1C2C2E&color=C96329
[python-badge]: https://img.shields.io/pypi/pyversions/zarrwhals?logo=python&labelColor=1C2C2E&color=3776AB
[license-badge]: https://img.shields.io/badge/License-MIT-yellow?labelColor=1C2C2E
[tests-badge]: https://img.shields.io/github/actions/workflow/status/srivarra/zarrwhals/test.yaml?branch=main&logo=github&label=tests&labelColor=1C2C2E
[build-badge]: https://img.shields.io/github/actions/workflow/status/srivarra/zarrwhals/build.yaml?branch=main&logo=github&label=build&labelColor=1C2C2E
[codecov-badge]: https://codecov.io/gh/srivarra/zarrwhals/graph/badge.svg?token=ST0ST1BTWU
[codspeed-badge]: https://img.shields.io/endpoint?url=https://codspeed.io/badge.json
[docs-badge]: https://img.shields.io/readthedocs/zarrwhals?logo=readthedocs&labelColor=1C2C2E
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[gitmoji-badge]: https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67?labelColor=1C2C2E

<!-- URLs -->
[pypi-url]: https://pypi.org/project/zarrwhals
[license-url]: https://opensource.org/licenses/MIT
[tests-url]: https://github.com/srivarra/zarrwhals/actions/workflows/test.yaml
[build-url]: https://github.com/srivarra/zarrwhals/actions/workflows/build.yaml
[codecov-url]: https://codecov.io/gh/srivarra/zarrwhals
[codspeed-url]: https://codspeed.io/srivarra/zarrwhals
[docs-url]: https://zarrwhals.readthedocs.io
[ruff-url]: https://github.com/astral-sh/ruff
[uv-url]: https://github.com/astral-sh/uv
[gitmoji-url]: https://gitmoji.dev/

Dataframe-agnostic Zarr storage powered by Narwhals

## Overview

Zarrwhals provides a simple API for storing DataFrames in Zarr format, using [Narwhals](https://narwhals-dev.github.io/narwhals/) for DataFrame interoperability. Inspired by [AnnData](https://github.com/scverse/anndata)'s approach to storing obs/var DataFrames as Zarr Stores.

## Quick Start

```python
import pandas as pd
import zarrwhals as zw

# Create a DataFrame
df = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "value": [10.5, 20.3, 30.1],
    "category": pd.Categorical(["A", "B", "A"])
})

# Write to Zarr
zw.to_zarr(df, "data.zarr")

# Read back
df_loaded = zw.from_zarr("data.zarr")
```

### With Polars

```python
import polars as pl
import zarrwhals as zw

# Write polars DataFrame
df_pl = pl.DataFrame({
    "a": [1, 2, 3],
    "b": ["x", "y", "z"]
})
zw.to_zarr(df_pl, "data.zarr")

# Read as pandas
df_pd = zw.from_zarr("data.zarr", backend="pandas")

# Or read back as polars
df_pl2 = zw.from_zarr("data.zarr", backend="polars")
```

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install zarrwhals:

<!--
1) Install the latest release of `zarrwhals` from [PyPI][]:

```bash
pip install zarrwhals
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/srivarra/zarrwhals.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/srivarra/zarrwhals/issues
[documentation]: https://zarrwhals.readthedocs.io
[changelog]: https://zarrwhals.readthedocs.io/en/latest/changelog.html
[api documentation]: https://zarrwhals.readthedocs.io/en/latest/api.html
