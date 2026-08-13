import numpy as np
from pandas import DataFrame, Series

from src.data_manipulation.utils.internal_checkers import ensure_data_has_one_column


def create_filtered_phases_mask(data: Series | DataFrame, filtered_data: Series | DataFrame) -> Series:
    """Create a boolean mask for filtered phases.

    Args:
        data (Series | DataFrame): The original data.
        filtered_data (Series | DataFrame): The filtered data.

    Returns:
        Series: A boolean mask indicating the filtered phases.

    Raises:
        ValueError: If the 'filtered_data' index is not a subset of the 'data' index or if the values in 'filtered_data'
         do not match the corresponding values in 'data' or data is empty but filtered_data is not.
    """
    _validate_params_values(data=data, filtered_data=filtered_data)

    if data.empty and filtered_data.empty:
        return Series(dtype=bool)

    if not data.empty and filtered_data.empty:
        return Series(False, index=data.index)

    mask = Series(False, index=data.index)
    mask.loc[filtered_data.index] = True

    return mask


def _validate_params_values(data: Series | DataFrame, filtered_data: Series | DataFrame) -> None:
    ensure_data_has_one_column(data=data)
    ensure_data_has_one_column(data=filtered_data)

    if not data.empty and not filtered_data.empty:
        if data.index.intersection(filtered_data.index).empty:
            raise ValueError("The 'filtered_data' index must be a subset of the 'data' index!")

        if not np.array_equal(filtered_data.values.flatten(), data.loc[filtered_data.index].values.flatten()):
            raise ValueError(
                "The 'filtered_data' values must match the corresponding values in 'data' for the same index!"
            )

    if data.empty and not filtered_data.empty:
        raise ValueError(
            "The 'data' is empty but 'filtered_data' is not empty! "
            "The 'filtered_data' index must be a subset of the 'data' index!"
        )
