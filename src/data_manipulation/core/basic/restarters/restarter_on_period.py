import dataclasses

from pandas import DataFrame, Series, Timedelta

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series, convert_series2frame
from src.data_manipulation.core.basic.index_processing.index_freq_inferrer import infer_index_freq
from src.data_manipulation.utils.internal_checkers import (
    ensure_data_has_one_column,
    ensure_positive_freq,
)


@dataclasses.dataclass
class RestartCumsumOnPeriodResult:
    restarted_cumsum: Series
    restarts_mask: Series


def restart_cumsum_on_period(data: Series | DataFrame, period: str | Timedelta) -> RestartCumsumOnPeriodResult:
    """
    Restarts the cumulative sum of the input data at regular intervals defined by the period.

    Args:
        data (Series | DataFrame): The input data with a datetime index.
        period (str | Timedelta): The interval at which the cumulative sum should restart.

    Returns:
        RestartCumsumOnPeriodResult:
            - restarted_cumsum: The restarted cumulative sum as a Series.
            - restarts_mask: A boolean mask indicating the restart points.

    Raises:
        ValueError: If the input values are invalid.
    """
    _validate_params(data=data, period=period)

    if data.empty:
        return RestartCumsumOnPeriodResult(restarted_cumsum=convert_frame2series(data=data), restarts_mask=Series())

    if len(data) == 1:
        return RestartCumsumOnPeriodResult(
            restarted_cumsum=convert_frame2series(data=data), restarts_mask=Series(False, index=data.index)
        )

    data = data.copy()

    df = convert_series2frame(data=data)
    stand_alone_col = df.columns[0]

    period = Timedelta(period)

    window = df.index.floor(period)
    restarted_cumsum = df[stand_alone_col].groupby(window).cumsum()

    df["downsampled_col"] = df[stand_alone_col].asfreq(freq=period)
    restarts_mask = df["downsampled_col"].notna()
    restarts_mask = Series(restarts_mask, index=df.index)

    return RestartCumsumOnPeriodResult(restarted_cumsum=restarted_cumsum, restarts_mask=restarts_mask)


def _validate_params(data: Series | DataFrame, period: str | Timedelta) -> None:
    ensure_data_has_one_column(data=data)
    ensure_positive_freq(freq=period)

    if not data.empty and len(data) > 1:
        index_freq = infer_index_freq(index=data.index)

        if index_freq is None:
            raise ValueError("Could not infer the frequency of the data index!")

        if Timedelta(period) <= Timedelta(index_freq):
            raise ValueError("The 'period' must be greater than the data index frequency!")
