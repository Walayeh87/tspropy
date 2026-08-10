import dataclasses
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series
from src.data_manipulation.utils.internal_checkers import ensure_data_has_one_column

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RestartCumsumOnThresholdResult:
    restarted_cumsum: Series
    restarts_mask: Series


def restart_cumsum_on_threshold(data: DataFrame | Series, threshold: int | float) -> RestartCumsumOnThresholdResult:
    ensure_data_has_one_column(data=data)

    data = data.copy()

    if data.empty:
        return RestartCumsumOnThresholdResult(
            restarted_cumsum=convert_frame2series(data=data),
            restarts_mask=Series(index=data.index),
        )

    series = convert_frame2series(data=data)

    if threshold <= 0:
        raise ValueError("The 'threshold' must be a positive number!")

    if threshold > series.sum():
        raise ValueError(
            f"The 'threshold' must be less than or equal to the sum of the data! "
            f"Data sum: {series.sum()}, threshold: {threshold}"
        )

    data_as_array = np.asarray(series, dtype=np.float32)

    restarted_cumsum = np.empty_like(data_as_array)
    restarts_mask = np.zeros(len(data_as_array), dtype=bool)

    current = 0.0
    res = restarted_cumsum
    flags = restarts_mask

    for index, value in enumerate(data_as_array):
        if np.isnan(value):
            res[index] = np.nan
            flags[index] = False
            continue

        test_sum = current + value

        if test_sum > threshold:
            current = value
            flags[index] = True
        else:
            current = test_sum

        res[index] = current

    return RestartCumsumOnThresholdResult(
        restarted_cumsum=Series(restarted_cumsum, index=data.index),
        restarts_mask=Series(restarts_mask, index=data.index),
    )


if __name__ == "__main__":
    freq = "1min"
    n = 100

    # time index
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)

    # create positive / negative phases
    phases = np.repeat([1, 1, -1, 1, 1], [20, 15, 25, 20, 20])

    # base signal
    signal = phases * (2 + 0.5 * np.random.randn(n))

    df = DataFrame({"value": signal}, index=idx)
    df.loc[df.index[10], "value"] = np.nan

    result = restart_cumsum_on_threshold(data=df["value"], threshold=10)
