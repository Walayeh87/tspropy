import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.mask_creators.filtered_phases_mask_creator import create_filtered_phases_mask
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


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
            Series([], index=pd.to_datetime([])),
        ),
    ],
)
def test_with_empty_params(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    restarted_cumsum = create_filtered_phases_mask(data=data, filtered_data=filtered_data)

    assert restarted_cumsum.equals(Series(dtype=bool))


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
            Series([5], index=pd.to_datetime(["2024-01-01"])),
        ),
    ],
)
def test_data_is_empty_but_filtered_is_not(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    with pytest.raises(ValueError):
        create_filtered_phases_mask(data=data, filtered_data=filtered_data)


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            Series([5, 6, 7], index=pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"])),
            DataFrame(data={"value": [5, 7]}, index=pd.to_datetime(["2024-01-01", "2024-01-03"])),
        ),
    ],
)
def test_data_with_mismatched_indexes(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    with pytest.raises(ValueError):
        create_filtered_phases_mask(data=data, filtered_data=filtered_data)


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            Series([5], index=pd.to_datetime(["2024-01-01"])),
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
        ),
        (
            Series(7, index=pd.to_datetime(["2024-01-01"])),
            DataFrame(data={"value": []}, index=pd.to_datetime([])),
        ),
    ],
)
def test_with_filtered_is_empty_but_data_not(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    restarted_cumsum = create_filtered_phases_mask(data=data, filtered_data=filtered_data)

    assert restarted_cumsum.equals(Series(False, index=data.index))


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            Series([5], index=pd.to_datetime(["2024-01-01"])),
            DataFrame(
                data={"value": [5, 6, 7], "other_value": [1, 2, 3]},
                index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ),
        ),
        (
            DataFrame(
                data={"value": [5, 6, 7], "other_value": [1, 2, 3]},
                index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            ),
            Series([5], index=pd.to_datetime(["2024-01-01"])),
        ),
    ],
)
def test_with_df_and_filtered_df_with_more_than_one_column(
    data: Series | DataFrame, filtered_data: Series | DataFrame
) -> None:
    with pytest.raises(DataframeDimensionError):
        create_filtered_phases_mask(data=data, filtered_data=filtered_data)


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            Series([5, 6, 7], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            DataFrame(data={"value": [5, 7]}, index=pd.to_datetime(["2024-01-01", "2024-01-03"])),
        ),
    ],
)
def test_scenario_1(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    restarted_cumsum = create_filtered_phases_mask(data=data, filtered_data=filtered_data)

    assert restarted_cumsum.equals(Series([True, False, True], index=data.index))


@pytest.mark.parametrize(
    "data, filtered_data",
    [
        (
            Series([5, 6, 7], index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            DataFrame(data={"value": [5, 6]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])),
        ),
    ],
)
def test_scenario_2(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    restarted_cumsum = create_filtered_phases_mask(data=data, filtered_data=filtered_data)

    assert restarted_cumsum.equals(Series([True, True, False], index=data.index))
