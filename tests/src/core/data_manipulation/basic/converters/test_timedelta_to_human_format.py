import pytest
from pandas import NaT, Timedelta

from src.core.data_manipulation.basic.converters.timedelta_to_human_format import convert_timedelta_to_human_format


@pytest.mark.parametrize(
    "timedelta, expected_result",
    [
        (Timedelta(days=2, hours=5, minutes=30, seconds=5, milliseconds=7), "2d 5h 30min 5s"),
        (Timedelta(hours=3, minutes=45), "3h 45min"),
        (Timedelta(minutes=15), "15min"),
        (Timedelta(minutes=-15), "-1d 23h 45min"),
        (Timedelta(minutes=0), "0min"),
        (Timedelta(seconds=0), "0min"),
    ],
)
def test_convert_timedelta_to_human_format(timedelta: Timedelta, expected_result: str) -> None:
    result = convert_timedelta_to_human_format(timedelta=timedelta)

    assert result == expected_result


def test_convert_timedelta_to_human_format_with_nat() -> None:
    with pytest.raises(ValueError):
        convert_timedelta_to_human_format(timedelta=NaT)
