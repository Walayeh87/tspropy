import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, DatetimeIndex, Series

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series
from src.data_manipulation.core.basic.mask_processing.mask_properties import get_mask_on_starts
from src.data_manipulation.core.basic.restarters.restarter_on_mask_starts import restart_cumsum_on_mask_starts
from src.data_manipulation.utils.custom_errors import (
    DataframeDimensionError,
    EmptyDataError,
    NonBooleanSeriesError,
    NonMatchingIndexesError,
)


def create_df(start: str, end: str, freq: str, value: int | float, nan_positions: list[int] | None = None) -> DataFrame:
    idx = pd.date_range(start=start, end=end, freq=freq)
    df = DataFrame({"value": value}, index=idx)
    if isinstance(nan_positions, list):
        df.iloc[nan_positions, 0] = np.nan

    return df


def create_series(
    start: str, end: str, freq: str, value: int | float, nan_positions: list[int] | None = None
) -> Series:
    idx = pd.date_range(start=start, end=end, freq=freq)
    series = Series(value, index=idx)
    if isinstance(nan_positions, list):
        series.iloc[nan_positions] = np.nan

    return series


def create_mask_starts(
    start: str, end: str, freq: str, value: bool, true_positions: list[int] | None = None
) -> Series[bool]:
    idx = pd.date_range(start=start, end=end, freq=freq)
    mask = Series(value, index=idx, dtype=bool)
    if isinstance(true_positions, list):
        mask.iloc[true_positions] = True

    mask_starts = get_mask_on_starts(mask=mask)

    return mask_starts


@pytest.mark.parametrize(
    "data, mask_starts",
    [
        (
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
            Series([], index=pd.to_datetime([])),
        ),
        (
            DataFrame(),
            Series(),
        ),
    ],
)
def test_with_empty_params(data: Series | DataFrame, mask_starts: Series) -> None:
    restarted_cumsum = restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)

    assert restarted_cumsum.equals(convert_frame2series(data=data))


@pytest.mark.parametrize(
    "data, mask_starts",
    [
        (
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
            Series([True], index=pd.to_datetime(["2020-10-10"])),
        ),
        (
            DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"])),
            Series([], index=pd.to_datetime([])),
        ),
    ],
)
def test_with_empty_data_or_mask(data: Series | DataFrame, mask_starts: Series) -> None:
    with pytest.raises(EmptyDataError):
        restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)


def test_with_non_boolean_mask() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    mask_starts = Series([3], index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(NonBooleanSeriesError):
        restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)


def test_with_non_matching_indexes() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    mask_starts = Series([True], index=pd.to_datetime(["2020-10-15"]))
    with pytest.raises(NonMatchingIndexesError):
        restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)


def test_with_df_of_more_than_1_col() -> None:
    data = DataFrame(data={"value1": [1], "value2": [7]}, index=pd.to_datetime(["2020-10-10"]))
    mask_starts = Series([True], index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(DataframeDimensionError):
        restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)


def test_when_data_has_one_row_and_mask_is_true() -> None:
    data = DataFrame([1], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    mask_starts = Series([True], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    restarted_cumsum = restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)

    assert isinstance(restarted_cumsum, Series)

    assert restarted_cumsum.index.equals(data.index)

    expected_index = restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1],
        name=restarted_cumsum.name,
        index=expected_index,
    )
    assert restarted_cumsum.equals(expected_restarted_cumsum)


def test_when_data_has_one_row_and_mask_is_false() -> None:
    data = DataFrame([1], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    mask_starts = Series([False], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    restarted_cumsum = restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)

    assert isinstance(restarted_cumsum, Series)

    assert restarted_cumsum.index.equals(data.index)

    expected_index = restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1],
        name=restarted_cumsum.name,
        index=expected_index,
    )
    assert restarted_cumsum.equals(expected_restarted_cumsum)


def test_with_no_gaps() -> None:
    start = "2020-01-01 00:00:00"
    end = "2020-01-01 00:15:00"
    freq = "1min"

    data = create_df(start=start, end=end, freq=freq, value=1)
    mask_starts = create_mask_starts(start=start, end=end, freq=freq, value=False, true_positions=[0, 5, 9])
    restarted_cumsum = restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)

    assert isinstance(restarted_cumsum, Series)

    assert restarted_cumsum.index.equals(data.index)

    expected_index = restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7],
        name=restarted_cumsum.name,
        index=expected_index,
    )
    assert restarted_cumsum.equals(expected_restarted_cumsum)


def test_with_gaps() -> None:
    start = "2020-01-01 00:00:00"
    end = "2020-01-01 00:15:00"
    freq = "1min"

    data = create_df(start=start, end=end, freq=freq, value=1, nan_positions=[2, 12])
    mask_starts = create_mask_starts(start=start, end=end, freq=freq, value=False, true_positions=[0, 5, 9])
    restarted_cumsum = restart_cumsum_on_mask_starts(data=data, mask_starts=mask_starts)

    assert isinstance(restarted_cumsum, Series)

    assert restarted_cumsum.index.equals(data.index)

    expected_index = restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1, 2, np.nan, 3, 4, 1, 2, 3, 4, 1, 2, 3, np.nan, 4, 5, 6],
        name=restarted_cumsum.name,
        index=expected_index,
    )

    assert restarted_cumsum.equals(expected_restarted_cumsum)
