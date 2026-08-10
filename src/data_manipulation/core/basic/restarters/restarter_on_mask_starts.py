import logging

from pandas import DataFrame, Series

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series, convert_series2frame
from src.data_manipulation.utils.internal_checkers import (
    ensure_all_params_are_not_empty,
    ensure_boolean_series,
    ensure_data_has_one_column,
    ensure_matching_indexes,
)

logger = logging.getLogger(__name__)


def restart_cumsum_on_mask_starts(data: Series | DataFrame, mask_starts: Series[bool]) -> Series:
    """
    Restarts the cumulative sum of a given data series or dataframe column whenever a condition in the mask_starts
    series is met.

    Parameters:
        data (Series | DataFrame): The input data, either a pandas Series or a single-column DataFrame.
        mask_starts (Series[bool]): A boolean Series indicating where the cumulative sum should restart.

    Returns:
        Series: A pandas Series containing the restarted cumulative sum.
    """

    if data.empty and mask_starts.empty:
        return convert_frame2series(data=data)

    ensure_all_params_are_not_empty(params=[data, mask_starts])
    ensure_boolean_series(param=mask_starts)
    ensure_matching_indexes(index1=data.index, index2=mask_starts.index)
    ensure_data_has_one_column(data=data)

    data = data.copy()

    df = convert_series2frame(data=data)
    stand_alone_col = df.columns[0]

    df["mask_starts"] = mask_starts
    df["restarted_cumsum"] = df.groupby(df["mask_starts"].cumsum())[stand_alone_col].cumsum()

    return df["restarted_cumsum"]
