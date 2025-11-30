"""Pydantic models and metadata for Zarrwhals DataFrame storage.

Following the pydantic-zarr pattern to provide type-safe schema validation
and structure modeling for Zarrwhals stores. Also includes metadata creation
and validation utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import zarr
from pydantic import BaseModel, ConfigDict, Field
from pydantic_zarr.v3 import ArraySpec, GroupSpec
from zarr.storage import StoreLike

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

# ══════════════════════════════════════════════════════════════════════════════
# Encoding Constants
# ══════════════════════════════════════════════════════════════════════════════

#: DataFrame encoding type identifier
DATAFRAME_ENCODING_TYPE = "dataframe"
#: DataFrame encoding version
DATAFRAME_ENCODING_VERSION = "0.0.1"
#: Reserved name for index array
RESERVED_INDEX_NAME = "_index"
#: Set of all reserved names
RESERVED_NAMES = {RESERVED_INDEX_NAME}
#: Metadata key for encoding type
KEY_ENCODING_TYPE = "encoding-type"


class DataFrameGroupAttributes(BaseModel):
    """Attributes for root DataFrame group stored in .zattrs."""

    model_config = ConfigDict(populate_by_name=True)

    encoding_type: Literal["dataframe"] = Field(alias="encoding-type")
    encoding_version: str = Field(alias="encoding-version")
    column_order: list[str] = Field(alias="column-order")
    index_name: str = Field(alias="_index")


class CategoricalGroupAttributes(BaseModel):
    """Attributes for categorical/enum group.

    Supports categorical (dynamic) and enum (fixed) categories.
    """

    model_config = ConfigDict(populate_by_name=True)

    encoding_type: Literal["categorical", "enum"] = Field(alias="encoding-type")
    encoding_version: Literal["2.0.0"] = Field(alias="encoding-version", default="2.0.0")
    ordered: bool = False


class CategoricalGroupSpec(GroupSpec):
    """Pydantic spec for categorical/enum group with codes and categories arrays."""

    attributes: CategoricalGroupAttributes  # type: ignore[assignment]


class ColumnArrayAttributes(BaseModel):
    """Attributes for column arrays stored in .zattrs."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    encoding_type: str = Field(alias="encoding-type")
    narwhals_dtype: str | None = None


# Union type for DataFrame members: can be arrays or categorical groups
DataFrameColumnSpec = ArraySpec[ColumnArrayAttributes] | CategoricalGroupSpec


class DataFrameGroupSpec(GroupSpec[DataFrameGroupAttributes, DataFrameColumnSpec]):
    """Pydantic model for Zarrwhals DataFrame group with validation."""

    @classmethod
    def from_zarr(cls, group: zarr.Group) -> Self:
        """Load DataFrameGroupSpec from Zarr group.

        Parameters
        ----------
        group
            Zarr group containing DataFrame.

        Returns
        -------
        DataFrameGroupSpec
            Validated spec.

        Raises
        ------
        ValueError
            If structure invalid.
        """
        attrs_dict = dict(group.attrs)
        attrs = DataFrameGroupAttributes(**attrs_dict)

        if attrs.index_name not in group:
            msg = f"Index array '{attrs.index_name}' not found in group"
            raise ValueError(msg)

        index_member = group[attrs.index_name]
        if isinstance(index_member, zarr.Group):
            msg = f"Index '{attrs.index_name}' must be an array, got group"
            raise TypeError(msg)

        missing_columns = [col for col in attrs.column_order if col not in group]
        if missing_columns:
            msg = f"Columns in column-order not found in group: {missing_columns}"
            raise ValueError(msg)

        members = {}
        members[attrs.index_name] = ArraySpec.from_zarr(index_member)

        for col_name in attrs.column_order:
            member = group[col_name]
            if isinstance(member, zarr.Array):
                if len(member.shape) != 1:
                    msg = f"Column '{col_name}' must be 1D, got shape {member.shape}"
                    raise ValueError(msg)
                members[col_name] = ArraySpec.from_zarr(member)
            elif isinstance(member, zarr.Group):
                encoding_type = member.attrs.get("encoding-type")
                if encoding_type not in ("categorical", "enum"):
                    msg = f"Column '{col_name}' must be an array, categorical, or enum group, got group with encoding-type={encoding_type}"
                    raise ValueError(msg)
                members[col_name] = CategoricalGroupSpec.from_zarr(member)

        return cls(attributes=attrs, members=members)


def validate_dataframe_store(store: StoreLike | zarr.Group) -> DataFrameGroupSpec:
    """Validate Zarr store and return DataFrameGroupSpec.

    Parameters
    ----------
    store
        Path or Zarr group.

    Returns
    -------
    DataFrameGroupSpec
        Validated spec.

    Raises
    ------
    ValueError
        If structure invalid.
    FileNotFoundError
        If path doesn't exist.
    """
    if isinstance(store, zarr.Group):
        group = store
    else:
        group = zarr.open_group(store=store, mode="r")
    return DataFrameGroupSpec.from_zarr(group)


def get_dataframe_spec(store: str | zarr.Group | Path) -> DataFrameGroupSpec:
    """Get DataFrameGroupSpec from Zarr store (alias for validate_dataframe_store).

    Parameters
    ----------
    store
        Path or Zarr group.

    Returns
    -------
    DataFrameGroupSpec
        DataFrame spec.
    """
    return validate_dataframe_store(store)


def inspect_dataframe_store(store: str | zarr.Group | Path) -> dict:
    """Get human-readable structure info from DataFrame store.

    Parameters
    ----------
    store
        Path or Zarr group.

    Returns
    -------
    dict
        Structure info with encoding_version, columns, index_name, column_info.
    """
    spec = validate_dataframe_store(store)
    attrs = spec.attributes

    column_info = {}
    for col_name in attrs.column_order:
        col_spec = spec.members[col_name]
        # Handle both ArraySpec and CategoricalGroupSpec
        if isinstance(col_spec, CategoricalGroupSpec):
            # Categorical column - get info from attributes
            cat_attrs = col_spec.attributes
            column_info[col_name] = {
                "shape": None,  # Categorical groups don't have a single shape
                "dtype": None,
                "encoding_type": cat_attrs.encoding_type,
                "ordered": cat_attrs.ordered,
                "narwhals_dtype": "Categorical" if cat_attrs.encoding_type == "categorical" else "Enum",
            }
        else:
            narwhals_dtype = col_spec.attributes.narwhals_dtype if col_spec.attributes else None
            column_info[col_name] = {
                "shape": col_spec.shape,
                "dtype": col_spec.data_type,
                "encoding_type": col_spec.attributes.encoding_type if col_spec.attributes else None,
                "narwhals_dtype": str(narwhals_dtype) if narwhals_dtype else None,
            }

    index_array = spec.members[attrs.index_name]
    index_info = {
        "shape": index_array.shape,
        "dtype": index_array.data_type,
    }

    return {
        "encoding_version": attrs.encoding_version,
        "num_columns": len(attrs.column_order),
        "columns": attrs.column_order,
        "index_name": attrs.index_name,
        "index_info": index_info,
        "column_info": column_info,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Metadata Creation Utilities
# ══════════════════════════════════════════════════════════════════════════════


def create_dataframe_metadata(
    column_names: list[str],
    index_name: str,
) -> DataFrameGroupAttributes:
    """Create DataFrame group metadata.

    Parameters
    ----------
    column_names
        Column names in order.
    index_name
        Index array name.

    Returns
    -------
    DataFrameGroupAttributes
        DataFrame metadata model.
    """
    return DataFrameGroupAttributes(
        encoding_type=DATAFRAME_ENCODING_TYPE,
        encoding_version=DATAFRAME_ENCODING_VERSION,
        column_order=column_names,
        index_name=index_name,
    )


def validate_column_names(column_names: list[str]) -> None:
    """Validate DataFrame column names.

    Parameters
    ----------
    column_names
        Column names to validate.

    Raises
    ------
    ValueError
        If reserved names or duplicates found.
    """
    for name in column_names:
        if name in RESERVED_NAMES:
            msg = f"'{name}' is a reserved name and cannot be used as a column name"
            raise ValueError(msg)

    if len(column_names) != len(set(column_names)):
        duplicates = {name for name in column_names if column_names.count(name) > 1}
        msg = f"Column names must be unique. Found duplicates: {duplicates}"
        raise ValueError(msg)
