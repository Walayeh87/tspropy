import pandas as pd
import pytest
from pandas import DataFrame, Series, Timedelta

from src.core.data_manipulation.basic.mask_processing.mask_properties import (
    convert_mask_into_ones_and_negative_ones,
    convert_mask_into_ones_and_zeros,
    get_mask_cumulative_sum_numbers,
    get_mask_off_starts,
    get_mask_off_stops,
    get_mask_on_starts,
    get_mask_on_stops,
    get_mask_phase_durations,
    get_mask_phase_numbers,
)
from src.utils.custom_errors import DataframeDimensionError


def test_mask_phase_numbers_if_mask_does_not_contain_bool() -> None:
    with pytest.raises(TypeError):
        get_mask_phase_numbers(mask=Series([5]))


def test_get_mask_phase_numbers_if_mask_is_df_with_2_cols() -> None:
    with pytest.raises(DataframeDimensionError):
        get_mask_phase_numbers(mask=DataFrame(data={"a": [True], "b": [False]}))


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[1]),
        ),
        (
            Series(data=[False]),
            Series(data=[-1]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[1, 1, 1]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[-1, -1, -1]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[1, -1, 2, 2, -2]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[-1, -1, 1, 1, -2]),
        ),
    ],
)
def test_get_mask_phase_numbers(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_phase_numbers(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[1]),
        ),
        (
            Series(data=[False]),
            Series(data=[-1]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[1, 2, 3]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[-1, -2, -3]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[1, -1, 1, 2, -1]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[-1, -2, 1, 2, -1]),
        ),
    ],
)
def test_get_mask_cumulative_sum_numbers(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_cumulative_sum_numbers(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True], index=pd.to_datetime(["2020-10-10"])),
            Series(data=[pd.NaT], index=pd.to_datetime(["2020-10-10"])),
        ),
        (
            Series(data=[False], index=pd.to_datetime(["2020-10-10"])),
            Series(data=[pd.NaT], index=pd.to_datetime(["2020-10-10"])),
        ),
        (
            Series(
                data=[True, True, True],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12"]),
            ),
            Series(
                data=[Timedelta(days=3), Timedelta(days=3), Timedelta(days=3)],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12"]),
            ),
        ),
        (
            Series(
                data=[False, False, False],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12"]),
            ),
            Series(
                data=[Timedelta(days=-3), Timedelta(days=-3), Timedelta(days=-3)],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12"]),
            ),
        ),
        (
            Series(
                data=[True, False, True, True, False],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14"]),
            ),
            Series(
                data=[Timedelta(days=1), Timedelta(days=-1), Timedelta(days=2), Timedelta(days=2), Timedelta(days=-1)],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14"]),
            ),
        ),
        (
            Series(
                data=[False, False, True, True, False],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14"]),
            ),
            Series(
                data=[Timedelta(days=-2), Timedelta(days=-2), Timedelta(days=2), Timedelta(days=2), Timedelta(days=-1)],
                index=pd.to_datetime(["2020-10-10", "2020-10-11", "2020-10-12", "2020-10-13", "2020-10-14"]),
            ),
        ),
    ],
)
def test_get_mask_phase_durations(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_phase_durations(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[True]),
        ),
        (
            Series(data=[False]),
            Series(data=[False]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[True, False, False]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[False, False, False]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[True, False, True, False, False]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[False, False, True, False, False]),
        ),
    ],
)
def test_get_mask_on_starts(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_on_starts(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[True]),
        ),
        (
            Series(data=[False]),
            Series(data=[False]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[False, False, True]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[False, False, False]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[True, False, False, True, False]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[False, False, False, True, False]),
        ),
    ],
)
def test_get_mask_on_stops(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_on_stops(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[False]),
        ),
        (
            Series(data=[False]),
            Series(data=[True]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[False, False, False]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[True, False, False]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[False, True, False, False, True]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[True, False, False, False, True]),
        ),
    ],
)
def test_get_mask_off_starts(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_off_starts(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[False]),
        ),
        (
            Series(data=[False]),
            Series(data=[True]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[False, False, False]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[False, False, True]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[False, True, False, False, True]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[False, True, False, False, True]),
        ),
    ],
)
def test_get_mask_off_stops(mask: Series | DataFrame, expected_result: Series) -> None:
    result = get_mask_off_stops(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[1]),
        ),
        (
            Series(data=[False]),
            Series(data=[0]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[1, 1, 1]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[0, 0, 0]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[1, 0, 1, 1, 0]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[0, 0, 1, 1, 0]),
        ),
    ],
)
def test_convert_mask_into_ones_and_zeros(mask: Series | DataFrame, expected_result: Series) -> None:
    result = convert_mask_into_ones_and_zeros(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)


@pytest.mark.parametrize(
    "mask, expected_result",
    [
        (
            Series(),
            Series(),
        ),
        (
            Series(data=[True]),
            Series(data=[1]),
        ),
        (
            Series(data=[False]),
            Series(data=[-1]),
        ),
        (
            Series(data=[True, True, True]),
            Series(data=[1, 1, 1]),
        ),
        (
            Series(data=[False, False, False]),
            Series(data=[-1, -1, -1]),
        ),
        (
            Series(data=[True, False, True, True, False]),
            Series(data=[1, -1, 1, 1, -1]),
        ),
        (
            Series(data=[False, False, True, True, False]),
            Series(data=[-1, -1, 1, 1, -1]),
        ),
    ],
)
def test_convert_mask_into_ones_and_negative_ones(mask: Series | DataFrame, expected_result: Series) -> None:
    result = convert_mask_into_ones_and_negative_ones(mask=mask)

    pd.testing.assert_series_equal(result, expected_result, check_dtype=False)
