import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, DatetimeIndex, Series

from src.core.data_manipulation.basic.restarters.restarter_on_mask import (
    RestartCumsumOnMaskResult,
    restart_cumsum_on_mask,
)
from src.utils.custom_errors import (
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


def create_mask(
    start: str, end: str, freq: str, value: int | float, false_positions: list[int] | None = None
) -> Series:
    idx = pd.date_range(start=start, end=end, freq=freq)
    mask = Series(value, index=idx, dtype=bool)
    if isinstance(false_positions, list):
        mask.iloc[false_positions] = False

    return mask


@pytest.mark.parametrize(
    "data, mask",
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
def test_with_empty_params(data: Series | DataFrame, mask: Series) -> None:
    result = restart_cumsum_on_mask(data=data, mask=mask)

    assert isinstance(result, RestartCumsumOnMaskResult)
    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.empty
    assert result.restarts_mask.empty


def test_with_non_boolean_mask() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    mask = Series([3], index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(NonBooleanSeriesError):
        restart_cumsum_on_mask(data=data, mask=mask)


def test_with_non_matching_indexes() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    mask = Series([True], index=pd.to_datetime(["2020-10-15"]))
    with pytest.raises(NonMatchingIndexesError):
        restart_cumsum_on_mask(data=data, mask=mask)


def test_with_df_of_more_than_1_col() -> None:
    data = DataFrame(data={"value1": [1], "value2": [7]}, index=pd.to_datetime(["2020-10-10"]))
    mask = Series([True], index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(DataframeDimensionError):
        restart_cumsum_on_mask(data=data, mask=mask)


@pytest.mark.parametrize(
    "data, mask",
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
def test_with_empty_data_or_mask(data: Series | DataFrame, mask: Series) -> None:
    with pytest.raises(EmptyDataError):
        restart_cumsum_on_mask(data=data, mask=mask)


def test_when_data_has_one_row_and_mask_is_true() -> None:
    data = DataFrame([1], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    mask = Series([True], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    result = restart_cumsum_on_mask(data=data, mask=mask)

    assert isinstance(result.restarted_cumsum, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(data.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1.0],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [True],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    assert result.restarted_cumsum.equals(expected_restarted_cumsum)
    assert result.restarts_mask.equals(expected_restarts_mask)


def test_when_data_has_one_row_and_mask_is_false() -> None:
    data = DataFrame([1], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    mask = Series([False], index=DatetimeIndex(["2020-01-01 00:00:00"]))
    result = restart_cumsum_on_mask(data=data, mask=mask)

    assert isinstance(result.restarted_cumsum, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(mask.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [np.nan],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [False],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )

    assert result.restarted_cumsum.equals(expected_restarted_cumsum)
    assert result.restarts_mask.equals(expected_restarts_mask)


def test_with_no_gaps() -> None:
    start = "2020-01-01 00:00:00"
    end = "2020-01-01 00:15:00"
    freq = "1min"

    data = create_df(start=start, end=end, freq=freq, value=1)
    mask = create_mask(start=start, end=end, freq=freq, value=True, false_positions=[0, 5, 6, 9])

    result = restart_cumsum_on_mask(data=data, mask=mask)

    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(mask.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [np.nan, 1, 2, 3, 4, np.nan, np.nan, 1, 2, np.nan, 1, 2, 3, 4, 5, 6],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )

    assert result.restarted_cumsum.equals(expected_restarted_cumsum)
    assert result.restarts_mask.equals(expected_restarts_mask)


def test_with_gaps() -> None:
    start = "2020-01-01 00:00:00"
    end = "2020-01-01 00:15:00"
    freq = "1min"

    data = create_df(start=start, end=end, freq=freq, value=1, nan_positions=[2, 12])
    mask = create_mask(start=start, end=end, freq=freq, value=True, false_positions=[0, 5, 6, 9])

    result = restart_cumsum_on_mask(data=data, mask=mask)

    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(mask.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [np.nan, 1, np.nan, 2, 3, np.nan, np.nan, 1, 2, np.nan, 1, 2, np.nan, 3, 4, 5],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [False, True, False, False, False, False, False, True, False, False, True, False, False, False, False, False],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )

    assert result.restarted_cumsum.equals(expected_restarted_cumsum)
    assert result.restarts_mask.equals(expected_restarts_mask)
