import pandas as pd
import pytest
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.df_processing.data_updater import update_data
from src.data_manipulation.utils.custom_errors import NonMatchingColumnsError


@pytest.mark.parametrize(
    "data, slices, expected_updated_data",
    [
        (
            Series(),
            Series(),
            Series(),
        ),
        (
            Series(data=[1, 2], index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            Series(),
            Series(data=[1, 2], index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            Series(data=[1, 2], index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            Series(data=[9], index=pd.to_datetime(["2020-10-11"])),
            Series(data=[1, 9], index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            DataFrame(),
            DataFrame(),
            DataFrame(),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [], "B": []}),
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [7], "B": [8]}, index=pd.to_datetime(["2020-10-10"])),
            DataFrame(data={"A": [7, 2], "B": [8, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [7], "B": [8]}, index=pd.to_datetime(["2020-10-10"])),
            DataFrame(data={"A": [7, 2], "B": [8, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
    ],
)
def test_with_valid_entries(
    data: Series | DataFrame, slices: Series | DataFrame, expected_updated_data: Series | DataFrame
) -> None:
    updated_data = update_data(data=data, slices=slices)

    assert updated_data.equals(expected_updated_data)
    assert isinstance(updated_data, type(expected_updated_data))


@pytest.mark.parametrize(
    "data, slices",
    [
        (DataFrame(), Series()),
        (Series(), DataFrame()),
    ],
)
def test_with_invalid_types(data: Series | DataFrame, slices: Series | DataFrame) -> None:
    with pytest.raises(TypeError):
        update_data(data=data, slices=slices)


@pytest.mark.parametrize(
    "data, slices",
    [
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [1, 2]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"C": [1, 2]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
        ),
    ],
)
def test_with_non_matching_cols(data: Series | DataFrame, slices: Series | DataFrame) -> None:
    with pytest.raises(NonMatchingColumnsError):
        update_data(data=data, slices=slices)


@pytest.mark.parametrize(
    "data, slices",
    [
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-12", "2020-10-13"])),
        ),
        (
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=pd.to_datetime(["2020-10-10", "2020-10-11"])),
            DataFrame(data={"A": [1, 2], "B": [3, 4]}, index=[1, 2]),
        ),
    ],
)
def test_with_non_matching_indexes(data: Series | DataFrame, slices: Series | DataFrame) -> None:
    with pytest.raises(ValueError):
        update_data(data=data, slices=slices)
