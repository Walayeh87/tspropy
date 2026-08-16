import pytest
from pandas import DataFrame, Series

from src.core.data_manipulation.basic.converters.series_and_frame import convert_frame2series, convert_series2frame
from src.utils.custom_errors import DataframeDimensionError


# tests for convert_series2frame
def test_convert_series2frame_when_empty_df_passed() -> None:
    data = DataFrame()
    result = convert_series2frame(data=data)

    assert result.equals(data)


def test_convert_series2frame_when_empty_name() -> None:
    with pytest.raises(ValueError):
        convert_series2frame(data=Series([5]), name="")


def test_convert_series2frame_when_empty_series_passed() -> None:
    data = Series()
    result = convert_series2frame(data=data)

    assert isinstance(result, DataFrame)
    assert len(result) == 0
    assert len(result.columns) == 1
    assert result.columns[0] == 0


def test_convert_series2frame_when_empty_series_with_name_passed() -> None:
    data = Series()
    given_name = "given_name"
    result = convert_series2frame(data=data, name=given_name)

    assert isinstance(result, DataFrame)
    assert len(result) == 0
    assert len(result.columns) == 1
    assert result.columns[0] == given_name


def test_convert_series2frame_when_series_with_one_item_passed() -> None:
    data = Series([5])
    result = convert_series2frame(data=data)

    assert isinstance(result, DataFrame)
    assert len(result) == 1
    assert len(result.columns) == 1
    assert result.columns[0] == 0


def test_convert_series2frame_when_series_with_one_item_and_name_passed() -> None:
    data = Series([6])
    given_name = "given_name"
    result = convert_series2frame(data=data, name=given_name)

    assert isinstance(result, DataFrame)
    assert len(result) == 1
    assert len(result.columns) == 1
    assert result.columns[0] == given_name


def test_convert_series2frame_when_series_with_multiple_items_passed() -> None:
    data = Series([5, 6])
    result = convert_series2frame(data=data)

    assert isinstance(result, DataFrame)
    assert len(result) == 2
    assert len(result.columns) == 1
    assert result.columns[0] == 0


def test_convert_series2frame_when_series_with_multiple_items_and_name_passed() -> None:
    data = Series([6, 7])
    given_name = "given_name"
    result = convert_series2frame(data=data, name=given_name)

    assert isinstance(result, DataFrame)
    assert len(result) == 2
    assert len(result.columns) == 1
    assert result.columns[0] == given_name


# tests for convert_frame2series
def test_convert_frame2series_when_df_with_multiple_cols_is_passed() -> None:
    data = DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
    with pytest.raises(DataframeDimensionError):
        convert_frame2series(data=data)


def test_convert_frame2series_when_empty_series_passed() -> None:
    data = Series()
    result = convert_frame2series(data=data)

    assert result.equals(data)


def test_convert_frame2series_when_empty_df_passed() -> None:
    data = DataFrame()
    result = convert_frame2series(data=data)

    assert isinstance(result, Series)
    assert len(result) == 0
    assert result.empty

    data1 = DataFrame(columns=[0])  # in pandas, data1 is considered as an empty df, too
    result1 = convert_frame2series(data=data1)

    assert isinstance(result1, Series)
    assert len(result1) == 0
    assert result1.empty
    assert result1.name == 0
    assert result.equals(result1)


def test_convert_frame2series_when_series_with_one_item_passed() -> None:
    data = DataFrame(data=[5])
    result = convert_frame2series(data=data)

    assert isinstance(result, Series)
    assert len(result) == 1
    assert result.name == 0


def test_convert_frame2series_when_series_with_multiple_items_passed() -> None:
    data = DataFrame(data=[5, 6])
    result = convert_frame2series(data=data)

    assert isinstance(result, Series)
    assert len(result) == 2
    assert result.name == 0


def test_convert_frame2series_when_series_with_multiple_items_and_named_column_passed() -> None:
    given_name = "given_name"
    data = DataFrame(data={given_name: [5, 6]})
    result = convert_frame2series(data=data)

    assert isinstance(result, Series)
    assert len(result) == 2
    assert result.name == given_name
