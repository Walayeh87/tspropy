import pandas as pd
import pytest

from src.data_manipulation.core.basic.df_processing.df_splitters import split_df_into_list_of_phases


def test_if_phase_nr_col_not_in_df() -> None:
    with pytest.raises(ValueError):
        split_df_into_list_of_phases(df=pd.DataFrame(data={"A": [5], "B": [7]}), phase_nr_col="wrong_col_name")


def test_if_df_without_rows_is_passed() -> None:
    result = split_df_into_list_of_phases(df=pd.DataFrame(data={"A": [], "phase_nr": []}), phase_nr_col="phase_nr")

    assert isinstance(result, list)
    assert len(result) == 0


def test_if_valid_df_with_one_row_is_passed() -> None:
    df = pd.DataFrame(data={"value": [10], "phase_nr": [1]})

    result = split_df_into_list_of_phases(df=df, phase_nr_col="phase_nr")

    assert isinstance(result, list)
    assert len(result) == 1

    assert isinstance(result[0], pd.DataFrame)
    assert len(result[0]) == 1
    assert result[0]["value"].tolist() == [10]


def test_if_valid_df_is_passed() -> None:
    df = pd.DataFrame(data={"value": [10, 20, 30, 40, 50, 60], "phase_nr": [1, 1, 2, 2, 2, 3]})

    result = split_df_into_list_of_phases(df=df, phase_nr_col="phase_nr")

    assert isinstance(result, list)
    assert len(result) == 3

    assert all(isinstance(phase_df, pd.DataFrame) for phase_df in result)

    assert len(result[0]) == 2
    assert len(result[1]) == 3
    assert len(result[2]) == 1

    assert result[0]["value"].tolist() == [10, 20]
    assert result[1]["value"].tolist() == [30, 40, 50]
    assert result[2]["value"].tolist() == [60]
