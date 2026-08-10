import pandas as pd
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.df_processing.row_droppers import drop_rows_with_duplicated_indices
from src.data_manipulation.utils.custom_errors import NonMatchingColumnsError


def update_data(data: Series | DataFrame, slices: Series | DataFrame) -> Series | DataFrame:
    """Update a pandas Series or DataFrame by applying values from `slices` to matching indices.

    `slices` must have the same type as `data`. When indices overlap, values from
    `slices` replace those in `data`. Returns the updated Series/DataFrame.

    Returns a new Series or DataFrame; does not modify `data` or `slices` in place.

    Raises:
        NonMatchingColumnsError: If columns of data and slices do not match.
        TypeError: If data and slices are not of the same type (both Series or both DataFrame).
        ValueError: If slices index is not contained in data index.
    """

    _validate_param_types(data=data, slices=slices)
    _validate_param_values(data=data, slices=slices)

    if slices.empty:
        return data

    data = data.copy()
    slices = slices.copy()

    extended_data = pd.concat([data, slices], axis="index")
    updated_data = drop_rows_with_duplicated_indices(data=extended_data, index2keep="last")

    updated_data = updated_data.sort_index()

    return updated_data


def _validate_param_types(data: Series | DataFrame, slices: Series | DataFrame) -> None:
    if (
        isinstance(data, DataFrame)
        and isinstance(slices, Series)
        or isinstance(data, Series)
        and isinstance(slices, DataFrame)
    ):
        raise TypeError("data and slices must be of the same type, either pandas Series or DataFrame!")


def _validate_param_values(data: Series | DataFrame, slices: Series | DataFrame) -> None:
    if isinstance(data, DataFrame) and isinstance(slices, DataFrame) and not data.columns.equals(slices.columns):
        raise NonMatchingColumnsError("Columns of data and slices must match!")

    if not slices.index.isin(data.index).all():
        raise ValueError("slices index must be contained or equal to data index!")
