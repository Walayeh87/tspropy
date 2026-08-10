import pytest

from src.data_manipulation.custom_objects.phase_duration import PhaseDuration
from src.data_manipulation.utils.custom_errors import InvalidTimedeltaError


def test_with_invalid_time_delta() -> None:
    with pytest.raises(InvalidTimedeltaError):
        PhaseDuration(value="bla")


def test_with_negative_time_delta() -> None:
    with pytest.raises(InvalidTimedeltaError):
        PhaseDuration(value="-1min")


def test_valid_time_delta() -> None:
    phase_dur = PhaseDuration(value="30min")

    assert phase_dur.value == "30min"
