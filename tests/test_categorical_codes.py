"""Tests for categorical codes extraction."""

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl

from zarrwhals.zdtypes import get_categorical_codes


class TestCategoricalCodes:
    """Test pure Narwhals categorical codes implementation."""

    def test_pandas_basic_categorical(self):
        """Test basic pandas categorical encoding."""
        cat = pd.Categorical(["A", "B", "A", "C"], categories=["A", "B", "C"])
        s = nw.from_native(pd.Series(cat), series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, 1, 0, 2])
        assert np.array_equal(categories, ["A", "B", "C"])
        assert ordered is False
        assert codes.dtype == np.int32

    def test_pandas_missing_values(self):
        """Test missing values map to -1."""
        cat = pd.Categorical(["A", None, "B", None], categories=["A", "B"])
        s = nw.from_native(pd.Series(cat), series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, -1, 1, -1])
        assert np.array_equal(categories, ["A", "B"])
        assert ordered is False

    def test_pandas_unused_categories(self):
        """Test unused categories don't affect codes."""
        cat = pd.Categorical(["A", "A"], categories=["A", "B", "C", "D"])
        s = nw.from_native(pd.Series(cat), series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, 0])
        assert np.array_equal(categories, ["A", "B", "C", "D"])
        assert ordered is False

    def test_pandas_empty_categorical(self):
        """Test empty categorical series."""
        cat = pd.Categorical([], categories=["A", "B"])
        s = nw.from_native(pd.Series(cat), series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert len(codes) == 0
        assert np.array_equal(categories, ["A", "B"])
        assert ordered is False

    def test_pandas_ordered_categorical(self):
        """Test ordered categorical."""
        cat = pd.Categorical(["low", "high", "low"], categories=["low", "medium", "high"], ordered=True)
        s = nw.from_native(pd.Series(cat), series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, 2, 0])
        assert np.array_equal(categories, ["low", "medium", "high"])
        assert ordered is True

    def test_polars_enum(self):
        """Test polars Enum encoding."""
        s_pl = pl.Series(["X", "Y", "X", "Z"], dtype=pl.Enum(["X", "Y", "Z"]))
        s = nw.from_native(s_pl, series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, 1, 0, 2])
        assert np.array_equal(categories, ["X", "Y", "Z"])
        assert ordered is True  # Enum is always ordered

    def test_polars_categorical(self):
        """Test polars Categorical encoding."""
        s_pl = pl.Series(["A", "B", "A"], dtype=pl.Categorical)
        s = nw.from_native(s_pl, series_only=True)

        codes, _categories, ordered = get_categorical_codes(s)

        # Verify codes are valid integers
        assert codes.dtype == np.int32
        assert len(codes) == 3
        assert codes[0] == codes[2]  # "A" should have same code
        assert ordered is False

    def test_polars_enum_with_missing(self):
        """Test polars Enum with missing values."""
        s_pl = pl.Series(["X", None, "Y"], dtype=pl.Enum(["X", "Y", "Z"]))
        s = nw.from_native(s_pl, series_only=True)

        codes, categories, ordered = get_categorical_codes(s)

        assert np.array_equal(codes, [0, -1, 1])
        assert np.array_equal(categories, ["X", "Y", "Z"])
        assert ordered is True

    def test_matches_pandas_native(self):
        """Verify our implementation matches pandas native .cat.codes."""
        cat = pd.Categorical(["A", "B", None, "A", "C"], categories=["A", "B", "C", "D"])
        s_pd = pd.Series(cat)
        s_nw = nw.from_native(s_pd, series_only=True)

        # Our implementation
        codes_ours, _, _ = get_categorical_codes(s_nw)

        # Pandas native
        codes_pandas = s_pd.cat.codes.to_numpy().astype(np.int32)

        np.testing.assert_array_equal(codes_ours, codes_pandas)
