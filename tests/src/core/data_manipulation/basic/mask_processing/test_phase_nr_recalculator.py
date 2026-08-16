import pandas as pd
import pytest
from pandas import Series

from src.core.data_manipulation.basic.mask_processing.phase_nr_recalculator import recalculate_phase_nr


@pytest.mark.parametrize(
    "phase_nr",
    [
        (pd.Series([0, 0, 2, "2", 4, 5, 5])),
    ],
)
def test_with_invalid_types(phase_nr: Series) -> None:
    with pytest.raises(TypeError):
        recalculate_phase_nr(phase_nr_series=phase_nr)


@pytest.mark.parametrize(
    "phase_nr",
    [(pd.Series([4, 4, 2, 2]))],
)
def test_with_invalid_values(phase_nr: Series) -> None:
    with pytest.raises(ValueError):
        recalculate_phase_nr(phase_nr_series=phase_nr)


@pytest.mark.parametrize(
    "phase_nr",
    [(pd.Series())],
)
def test_if_empty(phase_nr: Series) -> None:
    new_phase_nr = recalculate_phase_nr(phase_nr_series=phase_nr)
    new_phase_nr.equals(phase_nr)


@pytest.mark.parametrize(
    "phase_nr",
    [(pd.Series([4]))],
)
def test_with_length_of_one(phase_nr: Series) -> None:
    new_phase_nr = recalculate_phase_nr(phase_nr_series=phase_nr)
    new_phase_nr.equals(pd.Series([1], index=phase_nr.index))


@pytest.mark.parametrize(
    "phase_nr",
    [(pd.Series([0, 0, 4, 4, 5, 5, 6, 9, 9, 9]))],
)
def test_normal_case(phase_nr: Series) -> None:
    new_phase_nr = recalculate_phase_nr(phase_nr_series=phase_nr)
    new_phase_nr.equals(pd.Series([1, 1, 2, 2, 3, 3, 4, 5, 5, 5], index=phase_nr.index))
