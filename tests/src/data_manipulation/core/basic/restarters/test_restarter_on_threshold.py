import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.restarters.restarter_on_threshold import (
    RestartCumsumOnThresholdResult,
    restart_cumsum_on_threshold,
)
from src.data_manipulation.utils.custom_errors import DataframeDimensionError


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


def test_with_empty_data() -> None:
    data = DataFrame(data={"a": []}, index=pd.to_datetime([]))

    result = restart_cumsum_on_threshold(data=data, threshold=10)

    assert isinstance(result, RestartCumsumOnThresholdResult)
    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    pd.testing.assert_series_equal(result.restarted_cumsum, Series([], index=data.index, name="a"), check_dtype=False)
    assert result.restarts_mask.equals(Series(index=data.index))


def test_with_data_of_one_value() -> None:
    data = DataFrame(data=[1], index=pd.to_datetime(["2020-10-10"]))

    result = restart_cumsum_on_threshold(data=data, threshold=0.5)

    assert isinstance(result, RestartCumsumOnThresholdResult)
    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    pd.testing.assert_series_equal(result.restarted_cumsum, Series([1], index=data.index), check_dtype=False)
    assert result.restarts_mask.equals(Series(True, index=data.index))


def test_when_threshold_is_larger_than_the_data_sum() -> None:
    data = DataFrame(data={"value": [1, 2, 3]}, index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12"]))
    with pytest.raises(ValueError):
        restart_cumsum_on_threshold(data=data, threshold=10)


def test_with_data_of_multiple_cols() -> None:
    data = DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"]))
    with pytest.raises(DataframeDimensionError):
        restart_cumsum_on_threshold(data=data, threshold=10)


def test_with_negative_threshold() -> None:
    data = DataFrame(data={"value": [1, -2]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"]))
    with pytest.raises(ValueError):
        restart_cumsum_on_threshold(data=data, threshold=-5)


def test_without_gaps() -> None:
    data = create_df(start="2020-01-01 00:00:00", end="2020-01-01 00:15:00", freq="1min", value=1)
    result = restart_cumsum_on_threshold(data=data, threshold=5)

    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(data.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [False, False, False, False, False, True, False, False, False, False, True, False, False, False, False, True],
        name=result.restarts_mask.name,
        index=expected_index,
    )
    pd.testing.assert_series_equal(result.restarted_cumsum, expected_restarted_cumsum, check_dtype=False)
    pd.testing.assert_series_equal(result.restarts_mask, expected_restarts_mask, check_dtype=False)


def test_with_gaps() -> None:
    data = create_df(
        start="2020-01-01 00:00:00", end="2020-01-01 00:15:00", freq="1min", value=1, nan_positions=[0, 5, 9]
    )
    result = restart_cumsum_on_threshold(data=data, threshold=5)

    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(data.index)

    expected_index = result.restarted_cumsum.index
    expected_restarted_cumsum = Series(
        [np.nan, 1, 2, 3, 4, np.nan, 5, 1, 2, np.nan, 3, 4, 5, 1, 2, 3],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    expected_restarts_mask = Series(
        [False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False],
        name=result.restarts_mask.name,
        index=expected_index,
    )
    pd.testing.assert_series_equal(result.restarted_cumsum, expected_restarted_cumsum, check_dtype=False)
    pd.testing.assert_series_equal(result.restarts_mask, expected_restarts_mask, check_dtype=False)
