import pandas as pd
from pandas import DataFrame, Series

from src.core.data_manipulation.basic.df_processing.row_droppers import drop_rows_with_duplicated_indices


def test_if_empty_series_is_passed() -> None:
    data = Series()

    result = drop_rows_with_duplicated_indices(data=data, index2keep="first")

    assert result.equals(Series())


def test_if_empty_df_is_passed() -> None:
    data = DataFrame()

    result = drop_rows_with_duplicated_indices(data=data, index2keep="first")

    assert result.equals(DataFrame())


def test_if_df_without_index_duplication_is_passed() -> None:
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}

    data_with_range_index = DataFrame(data=data, index=[0, 1, 2])
    result = drop_rows_with_duplicated_indices(data=data_with_range_index, index2keep="first")

    assert result.equals(data_with_range_index)

    data_with_dt_index = DataFrame(data=data, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    result = drop_rows_with_duplicated_indices(data=data_with_dt_index, index2keep="first")

    assert result.equals(data_with_dt_index)


def test_with_first_option_if_df_with_index_duplication_is_passed() -> None:
    data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}

    data_with_range_index = DataFrame(data=data, index=[0, 1, 1, 2])
    expected_result_range_index = DataFrame(data={"A": [1, 2, 4], "B": [5, 6, 8]}, index=[0, 1, 2])
    result = drop_rows_with_duplicated_indices(data=data_with_range_index, index2keep="first")

    assert result.equals(expected_result_range_index)

    data_with_dt_index = DataFrame(
        data=data, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    )
    expected_result_dt_index = DataFrame(
        data={"A": [1, 2, 4], "B": [5, 6, 8]}, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    result = drop_rows_with_duplicated_indices(data=data_with_dt_index, index2keep="first")

    assert result.equals(expected_result_dt_index)


def test_with_last_option_if_df_with_index_duplication_is_passed() -> None:
    data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}

    data_with_range_index = DataFrame(data=data, index=[0, 1, 1, 2])
    expected_result_range_index = DataFrame(data={"A": [1, 3, 4], "B": [5, 7, 8]}, index=[0, 1, 2])
    result = drop_rows_with_duplicated_indices(data=data_with_range_index, index2keep="last")

    assert result.equals(expected_result_range_index)

    data_with_dt_index = DataFrame(
        data=data, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    )
    expected_result_dt_index = DataFrame(
        data={"A": [1, 3, 4], "B": [5, 7, 8]}, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    )
    result = drop_rows_with_duplicated_indices(data=data_with_dt_index, index2keep="last")

    assert result.equals(expected_result_dt_index)


def test_with_false_option_if_df_with_index_duplication_is_passed() -> None:
    data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}

    data_with_range_index = DataFrame(data=data, index=[0, 1, 1, 2])
    expected_result_range_index = DataFrame(data={"A": [1, 4], "B": [5, 8]}, index=[0, 2])
    result = drop_rows_with_duplicated_indices(data=data_with_range_index, index2keep=False)

    assert result.equals(expected_result_range_index)

    data_with_dt_index = DataFrame(
        data=data, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    )
    expected_result_dt_index = DataFrame(
        data={"A": [1, 4], "B": [5, 8]}, index=pd.to_datetime(["2024-01-01", "2024-01-03"])
    )
    result = drop_rows_with_duplicated_indices(data=data_with_dt_index, index2keep=False)

    assert result.equals(expected_result_dt_index)
