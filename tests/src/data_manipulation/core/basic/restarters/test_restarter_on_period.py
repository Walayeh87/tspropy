import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series, Timedelta

from src.data_manipulation.core.basic.restarters.restarter_on_period import (
    RestartCumsumOnPeriodResult,
    restart_cumsum_on_period,
)
from src.data_manipulation.utils.custom_errors import DataframeDimensionError, InvalidFreqError


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


@pytest.mark.parametrize(
    "data, period",
    [
        (
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
            "15min",
        ),
        (
            Series(),
            "1s",
        ),
    ],
)
def test_with_empty_params(data: Series | DataFrame, period: str | Timedelta) -> None:
    result = restart_cumsum_on_period(data=data, period=period)

    assert isinstance(result, RestartCumsumOnPeriodResult)
    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.empty
    assert result.restarts_mask.empty


def test_with_df_of_more_than_1_col() -> None:
    data = DataFrame(data={"value1": [1], "value2": [7]}, index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(DataframeDimensionError):
        restart_cumsum_on_period(data=data, period="1min")


def test_when_data_has_one_row() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    result = restart_cumsum_on_period(data=data, period="15min")

    assert isinstance(result, RestartCumsumOnPeriodResult)
    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.equals(Series([1], index=pd.to_datetime(["2020-10-10"])))
    assert result.restarts_mask.equals(Series([False], index=pd.to_datetime(["2020-10-10"])))


def test_with_invalid_period() -> None:
    data = DataFrame(data={"value": [1]}, index=pd.to_datetime(["2020-10-10"]))
    with pytest.raises(InvalidFreqError):
        restart_cumsum_on_period(data=data, period="invalid_period")


def test_when_period_is_smaller_than_or_equal_to_data_freq() -> None:
    data = DataFrame(data={"value": [1, 2]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"]))
    with pytest.raises(ValueError):
        restart_cumsum_on_period(data=data, period="1D")


def test_with_valid_inputs() -> None:
    data = create_df(
        start="2020-01-01 00:00:00",
        end="2020-01-01 00:59:00",
        freq="1min",
        value=1,
        nan_positions=[0, 5, 9],
    )

    result = restart_cumsum_on_period(data=data, period="15min")

    assert isinstance(result.restarted_cumsum, Series)
    assert isinstance(result.restarts_mask, Series)

    assert result.restarted_cumsum.index.equals(data.index)
    assert result.restarts_mask.index.equals(data.index)

    expected_index = result.restarted_cumsum.iloc[0:15].index
    expected_series = Series(
        [np.nan, 1, 2, 3, 4, np.nan, 5, 6, 7, np.nan, 8, 9, 10, 11, 12],
        name=result.restarted_cumsum.name,
        index=expected_index,
    )
    assert result.restarted_cumsum.iloc[0:15].equals(expected_series)

    assert np.array_equal(result.restarted_cumsum.iloc[15:30], np.arange(1, 16))
    assert np.array_equal(result.restarted_cumsum.iloc[30:45], np.arange(1, 16))
    assert np.array_equal(result.restarted_cumsum.iloc[45:60], np.arange(1, 16))
