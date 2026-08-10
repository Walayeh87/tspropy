import pandas as pd
import pytest
from pandas import DatetimeIndex, Timedelta

from src.data_manipulation.core.basic.index_processing.index_freq_inferrer import (
    InferIndexFreqDefaults,
    infer_index_freq,
)
from src.data_manipulation.utils.custom_errors import InvalidFreqError


def create_dt_index(start: str = "2020-10-10", periods: int = 20, freq: str = "1h") -> DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq)


def create_dt_index_with_gaps(start: str = "2020-10-10", periods: int = 20, freq: str = "1h") -> DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq).delete([3, 7])


@pytest.mark.parametrize(
    "index, freq_tolerance, accepted_ratio, expected_index_freq",
    [
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance + 5,
            InferIndexFreqDefaults.accepted_ratio,
            None,
        ),
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance - 8,
            InferIndexFreqDefaults.accepted_ratio,
            None,
        ),
    ],
)
def test_infer_index_freq_with_invalid_freq_tolerance(
    index: DatetimeIndex,
    freq_tolerance: float | int,
    accepted_ratio: float | int,
    expected_index_freq: Timedelta,
) -> None:
    with pytest.raises(ValueError):
        infer_index_freq(index=index, freq_tolerance=freq_tolerance, accepted_ratio=accepted_ratio)


@pytest.mark.parametrize(
    "index, freq_tolerance, accepted_ratio, expected_index_freq",
    [
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance,
            InferIndexFreqDefaults.accepted_ratio + 100,
            None,
        ),
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance,
            InferIndexFreqDefaults.accepted_ratio - 100,
            None,
        ),
    ],
)
def test_infer_index_freq_with_accepted_ratio_tolerance(
    index: DatetimeIndex,
    freq_tolerance: float | int,
    accepted_ratio: float | int,
    expected_index_freq: Timedelta,
) -> None:
    with pytest.raises(ValueError):
        infer_index_freq(index=index, freq_tolerance=freq_tolerance, accepted_ratio=accepted_ratio)


@pytest.mark.parametrize(
    "index, freq_tolerance, accepted_ratio, round_offset, expected_index_freq",
    [
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance,
            InferIndexFreqDefaults.accepted_ratio,
            "invalid_offset",
            None,
        ),
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance,
            InferIndexFreqDefaults.accepted_ratio,
            "-5s",
            None,
        ),
        (
            create_dt_index(freq="1h"),
            InferIndexFreqDefaults.freq_tolerance,
            InferIndexFreqDefaults.accepted_ratio,
            "s",
            None,
        ),
    ],
)
def test_infer_index_freq_with_accepted_round_offset(
    index: DatetimeIndex,
    freq_tolerance: float | int,
    accepted_ratio: float | int,
    round_offset: str,
    expected_index_freq: Timedelta,
) -> None:
    with pytest.raises(InvalidFreqError):
        infer_index_freq(
            index=index, freq_tolerance=freq_tolerance, accepted_ratio=accepted_ratio, round_offset=round_offset
        )


@pytest.mark.parametrize(
    "index, expected_index_freq",
    [
        (
            create_dt_index(freq="1h"),
            Timedelta(hours=1),
        ),
        (
            create_dt_index(freq="30min"),
            Timedelta(minutes=30),
        ),
        (
            create_dt_index_with_gaps(freq="1h"),
            Timedelta(hours=1),
        ),
        (
            create_dt_index_with_gaps(freq="30min"),
            Timedelta(minutes=30),
        ),
    ],
)
def test_infer_index_freq_with_regular_indices(
    index: DatetimeIndex,
    expected_index_freq: Timedelta,
) -> None:
    index_freq = infer_index_freq(index)

    assert index_freq == expected_index_freq


def test_infer_index_freq_with_too_short_indices() -> None:
    index = pd.DatetimeIndex([])
    index_freq = infer_index_freq(index)

    assert index_freq is None


@pytest.mark.parametrize(
    "index, expected_freq",
    [
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:59:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 03:00:00",
                    "2020-01-01 04:00:00",
                ]
            ),
            Timedelta(hours=1),
        ),
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:59",
                    "2020-01-01 00:01:00",
                    "2020-01-01 00:02:00",
                    "2020-01-01 00:03:00",
                    "2020-01-01 00:04:00",
                ]
            ),
            Timedelta(minutes=1),
        ),
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:01.01",
                    "2020-01-01 00:00:02.02",
                    "2020-01-01 00:00:03.006",
                    "2020-01-01 00:00:04",
                ]
            ),
            Timedelta(seconds=1),
        ),
    ],
)
def test_infer_index_freq_with_slightly_irregular_indices(
    index: DatetimeIndex,
    expected_freq: Timedelta,
) -> None:
    index_freq = infer_index_freq(index, freq_tolerance=0.49)

    assert index_freq == expected_freq


@pytest.mark.parametrize(
    "chaotic_index, expected_result",
    [
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 02:15:00",
                    "2020-01-01 01:10:00",
                    "2020-01-01 04:30:00",
                    "2020-01-01 03:05:00",
                    "2020-01-02 08:05:00",
                    "2020-01-07 09:05:00",
                ]
            ),
            None,
        )
    ],
)
def test_infer_index_freq_with_chaotic_index(chaotic_index: DatetimeIndex, expected_result: None) -> None:
    index_freq = infer_index_freq(chaotic_index)

    assert index_freq == expected_result


@pytest.mark.parametrize(
    "index_with_nat, expected_freq",
    [
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    pd.NaT,
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    pd.NaT,
                ]
            ),
            Timedelta(hours=1),
        )
    ],
)
def test_infer_index_freq_with_some_nat(
    index_with_nat: DatetimeIndex,
    expected_freq: Timedelta,
) -> None:
    index_freq = infer_index_freq(index=index_with_nat)

    assert index_freq == expected_freq


@pytest.mark.parametrize(
    "nat_index, expected_freq",
    [
        (
            pd.DatetimeIndex(
                [
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                    pd.NaT,
                ]
            ),
            None,
        )
    ],
)
def test_infer_index_freq_with_nat_index(
    nat_index: DatetimeIndex,
    expected_freq: Timedelta,
) -> None:
    index_freq = infer_index_freq(nat_index)

    assert index_freq == expected_freq


@pytest.mark.parametrize(
    "index, expected_freq",
    [
        (
            pd.DatetimeIndex(
                [
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                    "2020-01-01 00:00:00",
                ]
            ),
            None,
        )
    ],
)
def test_infer_index_freq_with_repetitive_timestamp(
    index: DatetimeIndex,
    expected_freq: Timedelta,
) -> None:
    index_freq = infer_index_freq(index)

    assert index_freq == expected_freq
