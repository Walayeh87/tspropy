import pytest

from src.core.data_manipulation.basic.list_processing.list_processing import convert_to_flat_list, is_sublist


@pytest.mark.parametrize(
    "sub, main, expected_result",
    [
        ([1, 2], [1, 2, 3], True),
        ([2, 3], [1, 2, 3], True),
        ([1, 3], [1, 2, 3], True),
        ([1, 5], [1, 2, 3], False),
        ([], [1, 2, 3], True),
        ([], [], True),
        ([1, 2, 3, 4], [1, 2, 3], False),
    ],
)
def test_is_sublist(sub: list, main: list, expected_result: bool) -> None:
    assert is_sublist(sub, main) == expected_result


@pytest.mark.parametrize(
    "list_of_lists, expected_result",
    [
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([[1], [2], [3]], [1, 2, 3]),
        ([], []),
        ([[1, 2, 3]], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
    ],
)
def test_convert_to_flat_list(list_of_lists: list, expected_result: list) -> None:
    assert convert_to_flat_list(list_of_lists) == expected_result
