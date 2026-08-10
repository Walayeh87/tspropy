from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp
from pandas.tseries.frequencies import to_offset

from src.data_manipulation.utils.custom_errors import (
    DataframeDimensionError,
    EmptyDataError,
    InvalidFreqError,
    InvalidTimedeltaError,
    InvalidTimestampError,
    NonBooleanSeriesError,
    NonMatchingIndexesError,
)


def ensure_boolean_series(param: Series) -> None:
    if param.dtype != bool:
        raise NonBooleanSeriesError(f"Expected boolean Series, got Series with dtype '{param.dtype}'")


def ensure_data_has_one_column(data: Series | DataFrame) -> None:
    if isinstance(data, Series):
        return

    if isinstance(data, DataFrame) and len(data.columns) != 1:
        raise DataframeDimensionError(
            f"Expected DataFrame with one column, got DataFrame with {len(data.columns)} columns"
        )


def ensure_matching_indexes(index1: DatetimeIndex, index2: DatetimeIndex) -> None:
    if not index1.equals(index2):
        raise NonMatchingIndexesError("The indexes do not match.")


def ensure_all_params_are_not_empty(params: list[Series | DataFrame]) -> None:
    all_full = all(not param.empty for param in params)

    if not all_full:
        raise EmptyDataError("All parameters must be non-empty.")


def ensure_positive_freq(freq: str | Timedelta) -> None:
    if isinstance(freq, Timedelta) and freq <= Timedelta("0"):
        raise InvalidFreqError(f"The 'freq' must be a positive Timedelta. Got '{freq}' instead.")

    if isinstance(freq, str):
        if not _is_valid_freq_as_str(freq_as_str=freq):
            raise InvalidFreqError("The 'freq' must be a valid pandas frequency string!")

        if Timedelta(freq) <= Timedelta("0"):
            raise InvalidFreqError(f"The 'freq' must be a positive Timedelta. Got '{freq}' instead.")


def ensure_valid_timestamp(timestamp: str | Timestamp) -> None:
    if isinstance(timestamp, Timestamp):
        return

    if isinstance(timestamp, str):
        if not _is_valid_timestamp_as_str(timestamp_as_str=timestamp):
            raise InvalidTimestampError("The 'timestamp' must be a valid pandas timestamp string!")


def ensure_positive_timedelta(timedelta: str | Timedelta) -> None:
    if isinstance(timedelta, Timedelta) and timedelta <= Timedelta(0):
        raise InvalidTimedeltaError(f"Timedelta must be positive. Got '{timedelta}' instead.")

    if isinstance(timedelta, str):
        if not _is_valid_timedelta_as_str(timedelta_as_str=timedelta):
            raise InvalidTimedeltaError("The 'timedelta' must be a valid pandas timedelta string!")

        if Timedelta(timedelta) <= Timedelta(0):
            raise InvalidTimedeltaError(f"Timedelta must be positive. Got '{timedelta}' instead.")


def _is_valid_freq_as_str(freq_as_str: str) -> bool:
    has_number = any(char.isdigit() for char in freq_as_str)
    if not has_number:
        return False

    try:
        to_offset(freq_as_str)
        return True
    except ValueError:
        return False


def _is_valid_timestamp_as_str(timestamp_as_str: str) -> bool:
    try:
        Timestamp(timestamp_as_str)
        return True
    except ValueError:
        return False


def _is_valid_timedelta_as_str(timedelta_as_str: str) -> bool:
    try:
        Timedelta(timedelta_as_str)
        return True
    except ValueError:
        return False
