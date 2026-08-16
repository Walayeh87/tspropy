import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DatetimeIndex, Timedelta, TimedeltaIndex

from src.utils.internal_checkers import ensure_positive_freq

logger = logging.getLogger(__name__)


@dataclass
class InferIndexFreqDefaults:
    freq_tolerance: float | int = 0.05
    accepted_ratio: float | int = 0.8
    round_offset: str = "1s"  # It must include a number. "s" is an invalid input!


def infer_index_freq(
    index: DatetimeIndex,
    freq_tolerance: float | int = InferIndexFreqDefaults.freq_tolerance,
    accepted_ratio: float | int = InferIndexFreqDefaults.accepted_ratio,
    round_offset: str = InferIndexFreqDefaults.round_offset,
) -> Timedelta | None:
    """Finds the frequency of a basically uniform/regular index by looking at the most frequent index delta. It can
    deal with irregular indexes and indexes with gaps.

    Parameters
    ----------

    index : The index of a pandas DataFrame or Series to inspect.
    freq_tolerance: The tolerated relative deviation from the median index delta in order to consider an index delta.
    accepted_ratio: The ratio of tolerated deltas to total deltas to consider the index as regular.
    round_offset: The offset alias to round the inferred frequency to. If empty, no rounding is applied.
    Rounding can be useful to avoid inferring weird frequencies.

    Returns
    -------

    Timedelta: The inferred frequency of the index, or None if the index is too short, chaotic, or includes only nans.
    """
    _validate_params(accepted_ratio=accepted_ratio, freq_tolerance=freq_tolerance, round_offset=round_offset)

    if len(index) <= 1:
        logger.warning(
            "Could not infer index frequency since the provided dataset is either empty or consists of only one"
            " row. None is returned"
        )
        return None

    index_without_nans = index.dropna()
    if len(index_without_nans) == 0:
        logger.warning("Could not infer index frequency since the index contains only NaN values. None is returned.")
        return None

    if not index_without_nans.is_monotonic_increasing:
        index_without_nans = index.sort_values()

    deltas = calculate_index_deltas(index=index_without_nans)
    mask_of_positive_deltas = deltas > pd.to_timedelta(0)
    no_positive_deltas = (~mask_of_positive_deltas).all()

    if no_positive_deltas:
        logger.warning(
            f"Could not infer index frequency since the index contains mainly one unique timestamp:"
            f" ''{index_without_nans[0]}''. None is returned."
        )
        return None

    positive_median = get_positive_median(deltas=deltas, mask_of_positive_deltas=mask_of_positive_deltas)
    mask_tolerated_median = get_mask_of_tolerated_median(
        deltas=deltas, freq_tolerance=freq_tolerance, positive_median=positive_median
    )

    total_deltas = mask_of_positive_deltas.sum()
    valid_deltas = mask_tolerated_median.sum()

    if valid_deltas < accepted_ratio * total_deltas:
        logger.warning("Could not infer index frequency since the index is a chaotic/irregular one. None is returned.")
        return None

    index_freq = deltas[mask_tolerated_median].mean()

    if round_offset == "":
        return index_freq

    return index_freq.round(freq=round_offset)


def _validate_params(accepted_ratio: float | int, freq_tolerance: float | int, round_offset: str) -> None:
    min_tolerance_allowed = 0
    max_tolerance_allowed = 1
    if not (min_tolerance_allowed <= freq_tolerance <= max_tolerance_allowed):
        raise ValueError(
            f"freq_tolerance must be between {min_tolerance_allowed} and {max_tolerance_allowed}. Got {freq_tolerance}."
        )

    min_ratio_allowed = 0.7
    max_ratio_allowed = 1
    if not (min_ratio_allowed <= accepted_ratio <= max_ratio_allowed):
        raise ValueError(
            f"accepted_ratio must be between {min_ratio_allowed} and {max_ratio_allowed}. Got {accepted_ratio}."
        )

    ensure_positive_freq(freq=round_offset)


def get_positive_median(deltas: TimedeltaIndex, mask_of_positive_deltas: np.ndarray) -> Timedelta:
    return deltas[mask_of_positive_deltas].median()


def get_mask_of_tolerated_median(
    deltas: TimedeltaIndex, freq_tolerance: float, positive_median: Timedelta
) -> np.ndarray:
    mask1 = deltas >= positive_median * (1 - freq_tolerance)
    mask2 = deltas <= positive_median * (1 + freq_tolerance)
    return mask1 & mask2


def calculate_index_deltas(index: DatetimeIndex) -> TimedeltaIndex:
    return index[1:] - index[:-1]
