from typing import Literal

from pandas import DataFrame, Series


def drop_rows_with_duplicated_indices(
    data: DataFrame | Series, index2keep: Literal["first", "last", False] = "first"
) -> Series | DataFrame:
    """Drop rows with duplicated indices from a DataFrame or Series.

    Args:
        data (DataFrame | Series): The input DataFrame or Series.
        index2keep (Literal["first", "last", False], optional): Which duplicate to keep. Defaults to "first".

    Returns:
        Series | DataFrame: The DataFrame or Series with duplicated indices dropped.
    """
    return data[~data.index.duplicated(keep=index2keep)]
