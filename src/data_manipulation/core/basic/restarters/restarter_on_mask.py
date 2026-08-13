import dataclasses
import logging

import numpy as np
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series, convert_series2frame
from src.data_manipulation.core.basic.mask_processing.mask_properties import get_mask_on_starts
from src.data_manipulation.utils.internal_checkers import (
    ensure_all_params_are_not_empty,
    ensure_boolean_series,
    ensure_data_has_one_column,
    ensure_matching_indexes,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RestartCumsumOnMaskResult:
    restarted_cumsum: Series
    restarts_mask: Series


def restart_cumsum_on_mask(data: Series | DataFrame, mask: Series[bool]) -> RestartCumsumOnMaskResult:
    if data.empty and mask.empty:
        return RestartCumsumOnMaskResult(
            restarted_cumsum=convert_frame2series(data=data),
            restarts_mask=convert_frame2series(data=mask),
        )

    ensure_all_params_are_not_empty(params=[data, mask])
    ensure_boolean_series(param=mask)
    ensure_matching_indexes(index1=data.index, index2=mask.index)
    ensure_data_has_one_column(data=data)

    data = data.copy()

    df = convert_series2frame(data=data)
    stand_alone_col = df.columns[0]

    df["mask"] = mask
    df["restarts_mask"] = get_mask_on_starts(mask=mask)
    df["restarted_cumsum"] = df.groupby(df["restarts_mask"].cumsum())[stand_alone_col].cumsum()
    df.loc[~mask, "restarted_cumsum"] = np.nan

    return RestartCumsumOnMaskResult(
        restarted_cumsum=df["restarted_cumsum"],
        restarts_mask=df["restarts_mask"],
    )
