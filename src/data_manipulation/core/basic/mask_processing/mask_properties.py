import functools
import logging
from enum import auto

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.data_manipulation.core.basic.converters.series_and_frame import convert_frame2series
from src.data_manipulation.core.basic.index_processing.index_freq_inferrer import infer_index_freq
from src.data_manipulation.utils.auto_names_modifier import AutoNamesModifier

logger = logging.getLogger(__name__)


class ColNames(AutoNamesModifier):
    INTS = auto()
    PHASE_NUMBERS = auto()


def validate_and_process_mask(func):
    """
    Decorator that validates and preprocesses boolean mask inputs for mask processing functions.

    This decorator ensures consistent preprocessing of mask inputs across all mask processing
    functions in this module. It implements a validation pipeline that:
      1. Converts DataFrame inputs to Series using `convert_frame2series()`
      2. Returns an empty Series copy if the input is empty (short-circuit)
      3. Validates that the dtype is boolean, raising TypeError if not
      4. Calls the decorated function with the validated Series

    The decorator preserves the original function's metadata (name, docstring, etc.)
    using functools.wraps, ensuring proper function introspection.

    Args:
        func (callable): A function that accepts a boolean mask Series as its first parameter
                         (after self, if a method). The function signature should be:
                         func(mask: Series | DataFrame, *args, **kwargs) -> Series

    Returns:
        callable: A wrapper function that performs validation and preprocessing before calling
                  the decorated function. The wrapper has the same signature as the input function
                  but with additional safety guarantees.

    Raises:
        TypeError: If the input mask (after DataFrame-to-Series conversion) does not have
                   boolean dtype. This is raised by the wrapper before the decorated function
                   is called.
        DataframeDimensionError: If the input is a DataFrame with more than one column.
                                 This is raised by `convert_frame2series()` during conversion.


    Example:
        >>> @validate_and_process_mask
        ... def get_mask_on_starts(mask: Series | DataFrame) -> Series:
        ...     # Function implementation
        ...     return result
        >>> # Now the function automatically validates input:
        >>> mask = Series([True, False, True])
        >>> result = get_mask_on_starts(mask)  # Succeeds
        >>> mask = Series([1, 0, 1])  # Integer mask
        >>> result = get_mask_on_starts(mask)  # Raises TypeError: "The mask must be boolean!"
        >>> mask = Series([])  # Empty boolean mask
        >>> result = get_mask_on_starts(mask)  # Returns empty Series copy without calling function

    Use cases:
      - Ensuring consistent input validation across all mask processing functions
      - Reducing code duplication in function implementations
      - Providing clear error messages for invalid inputs
      - Optimizing empty mask handling (early return)
      - Enabling seamless DataFrame and Series inputs throughout the API

    Notes:
      - This decorator is applied to most public mask processing functions in this module.
      - Functions decorated with this decorator do not need to implement their own
        DataFrame conversion or dtype validation logic.
      - The empty check happens before dtype validation, so empty inputs of any type
        (even non-boolean) will return an empty copy without raising TypeError.
      - The decorator modifies only validation and preprocessing; it does not alter
        the core logic or output of decorated functions.
      - When a DataFrame is converted to a Series, the conversion uses `convert_frame2series()`,
        which enforces single-column constraints.
      - Some internal helper functions (like _get_start_numbers, _get_stop_numbers) may not
        be decorated if they receive already-validated Series inputs from other decorated functions.
    """

    @functools.wraps(func)
    def wrapper(mask: Series | DataFrame, *args, **kwargs):
        mask = convert_frame2series(data=mask)

        if mask.empty:
            return mask.copy()

        if mask.dtype != bool:
            raise TypeError("The mask must be boolean!")

        return func(mask, *args, **kwargs)

    return wrapper


@validate_and_process_mask
def get_mask_phase_numbers(mask: Series | DataFrame) -> Series:
    """
    Assign unique phase numbers to consecutive sequences of True and False values in a boolean mask.

    This function identifies alternating phases (consecutive runs) of True and False values
    in the input mask and assigns each phase a unique number. Phases with True values receive
    positive cumulative integers, while phases with False values receive corresponding negative
    integers. This allows tracking and grouping of consecutive True/False sequences.

    Args:
        mask: A pandas Series or DataFrame containing boolean values. If a DataFrame is provided,
              it will be converted to a Series. The mask must have a boolean dtype (validated by
              the @validate_and_process_mask decorator).

    Returns:
        Series: A Series with the same index and length as the input mask, containing phase numbers.
                - Positive numbers (1, 2, 3, ...) indicate phases where the mask is True
                - Negative numbers (-1, -2, -3, ...) indicate phases where the mask is False
                - Each consecutive sequence (phase) receives a unique number

    Raises:
        TypeError: If the mask does not have a boolean dtype (raised by @validate_and_process_mask).
        DataframeDimensionError: If the input is a DataFrame with more than one column
         (raised by @validate_and_process_mask).

    Example:
        >>> mask = Series([True, True, False, False, True, True, True])
        >>> phase_numbers = get_mask_phase_numbers(mask)
        >>> # Result: [1, 1, -1, -1, 2, 2, 2]

    Note:
        - This function is decorated with @validate_and_process_mask which:
          * Converts DataFrame inputs to Series
          * Returns an empty Series copy if input is empty
          * Validates that the input has boolean dtype
        - The phase numbers are determined by cumulative sums of phase transitions.
        - Useful for identifying and grouping consecutive on/off periods in time series data.
    """
    mask_on_starts = get_mask_on_starts(mask=mask)
    mask_off_starts = get_mask_off_starts(mask=mask, exclude_first=True)

    start_numbers = _get_start_numbers(mask_on_starts=mask_on_starts)
    stop_numbers = _get_stop_numbers(mask_off_starts=mask_off_starts, first_mask_item=mask.iloc[0])

    start_numbers[~mask] = stop_numbers[~mask]

    # After the previous replacement, the start_numbers becomes phase_numbers
    phase_numbers = start_numbers

    return phase_numbers


@validate_and_process_mask
def get_mask_cumulative_sum_numbers(mask: Series | DataFrame) -> Series:
    """
    Compute a signed cumulative index inside each consecutive True/False phase of a boolean mask.

    For each consecutive run (phase) of equal boolean values in `mask`, this function returns a
    cumulative count that starts at 1 (or -1) and increments (or decrements) within the phase:
      - For a True-phase of length N the returned values are [1, 2, ..., N]
      - For a False-phase of length M the returned values are [-1, -2, ..., -M]

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).
            If a DataFrame is provided it will be converted to a Series by the
            `@validate_and_process_mask` decorator (via `convert_frame2series`).

    Returns:
        pandas.Series: Signed cumulative counts within each consecutive phase. The index matches
        the input mask's index. Positive integers correspond to positions inside True-phases,
        negative integers correspond to positions inside False-phases.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column
         (raised by `@validate_and_process_mask`).

    Examples:
        >>> mask = Series([True, True, False, False, False, True])
        >>> # Phase decomposition:
        >>> #  phase 1 (True) length 2 -> [1, 2]
        >>> #  phase 2 (False) length 3 -> [-1, -2, -3]
        >>> #  phase 3 (True) length 1 -> [1]
        >>> get_mask_cumulative_sum_numbers(mask)
        0    1
        1    2
        2   -1
        3   -2
        4   -3
        5    1
        dtype: int64#

        >>> mask = Series([False, True, False, False, False, True])
        >>> # Phase decomposition:
        >>> #  phase 1 (False) length 1 -> [-1]
        >>> #  phase 2 (True) length 1 -> [1]
        >>> #  phase 3 (False) length 3 -> [-1, -2, -3]
        >>> #  phase 4 (True) length 1 -> [1]
        >>> get_mask_cumulative_sum_numbers(mask)
        0   -1
        1    1
        2   -1
        3   -2
        4   -3
        5    1
        dtype: int64#

    Use cases:
      - numbering samples inside each on/off period (e.g., time-series event lengths)
      - creating per-phase relative positions for grouping, windowing, or aggregation
    """
    df = _get_df_of_phase_numbers_and_ints(mask=mask)

    mask_cumsum_numbers = df.groupby(by=ColNames.PHASE_NUMBERS)[ColNames.INTS].cumsum()
    mask_cumsum_numbers.name = None  # To remove the existing name "ints", which is incorrect ...

    return mask_cumsum_numbers


@validate_and_process_mask
def get_mask_phase_durations(mask: Series | DataFrame) -> Series:
    # TODO: this functions is not designed for irregular indexes. Consider improving it.
    """
    Calculate the duration of each consecutive True/False phase as Timedelta counts using the mask's index frequency.

    This function computes the length (duration) of each phase (consecutive run of True or False)
    in the provided boolean mask and returns a Series with the same index where each element is the
    duration of the phase that element belongs to.


    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).

    Returns:
        pandas.Series: Phase durations aligned with the input mask's index.
                       - If the index frequency is inferable, each value is the phase duration in
                         the index's frequency units (e.g. Timedelta, numeric depending on freq type).
                       - The sign of values follows `get_mask_phase_durations_as_ints`:
                         positive durations for True-phases, negative durations for False-phases.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column.

    Examples:
        >>> mask = Series([True, True, False, False, True], index=pd.date_range("2023-01-01", periods=5, freq="H"))
        >>> # Phases: True(2h), False(2h), True(1h)
        >>> get_mask_phase_durations(mask)
        2023-01-01 00:00:00   0 days 02:00:00
        2023-01-01 01:00:00   0 days 02:00:00
        2023-01-01 02:00:00  -1 day + 23:58:00
        2023-01-01 03:00:00  -1 day + 23:58:00
        2023-01-01 04:00:00   0 days 01:00:00
        dtype: timedelta64[ns]

        >>> # If the index has no inferable frequency (e.g. irregular timestamps), a Series of NaT is returned:
        >>> mask = Series([True, True], index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01") + pd.Timedelta(seconds=37)])
        >>> get_mask_phase_durations(mask)
        2023-01-01 00:00:00   NaT
        2023-01-01 01:00:00   NaT
        dtype: timedelta64[ns]

    Notes:
      - Use `get_mask_phase_durations_as_ints` if you need raw integer counts (number of samples per phase)
        instead of scaled durations.
    """
    phase_durations_as_ints = get_mask_phase_durations_as_ints(mask=mask)
    phase_durations_as_ints.name = None  # To remove the existing name "ints", which is incorrect ...

    index_freq = infer_index_freq(index=phase_durations_as_ints.index)
    if index_freq is not None:
        return phase_durations_as_ints * index_freq
    else:
        logger.warning("Index frequency cannot be inferred. Returning Series of NaT.")
        mask_phase_durations = Series(data=[pd.NaT] * len(phase_durations_as_ints), index=phase_durations_as_ints.index)
        return mask_phase_durations


@validate_and_process_mask
def get_mask_phase_durations_as_ints(mask: Series | DataFrame) -> Series:
    """
    Calculate the duration (length) of each consecutive True/False phase as integer counts.

    This function returns the number of consecutive samples in each phase of the input boolean mask.
    Each position in the output Series contains the length of the phase to which that position belongs.
    The value is positive for True-phases and negative for False-phases.

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).

    Returns:
        pandas.Series: Integer phase durations aligned with the input mask's index.
                       Each element contains the count of samples in its phase:
                       - Positive integers for positions inside True-phases
                       - Negative integers for positions inside False-phases
                       - All positions within the same phase have the same absolute value

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column

    Examples:
        >>> mask = Series([True, True, False, False, False, True])
        >>> # Phase decomposition:
        >>> #  True-phase of length 2 -> [2, 2]
        >>> #  False-phase of length 3 -> [-3, -3, -3]
        >>> #  True-phase of length 1 -> [1]
        >>> get_mask_phase_durations_as_ints(mask)
        0    2
        1    2
        2   -3
        3   -3
        4   -3
        5    1
        dtype: int64

        >>> mask = Series([False, True, False, False])
        >>> # False-phase (1), True-phase (1), False-phase (2)
        >>> get_mask_phase_durations_as_ints(mask)
        0   -1
        1    1
        2   -2
        3   -2
        dtype: int64

    Use cases:
      - Counting the number of samples per phase for statistical analysis
      - Creating phase-level metadata (how long each on/off period is)
      - Filtering or weighting samples based on phase lengths
      - Feeding into `get_mask_phase_durations` to convert counts to time-based durations

    Notes:
      - This is the "raw" (integer) version. Use `get_mask_phase_durations` if you need
        durations scaled by the index frequency (e.g., expressed in time units).
    """
    df = _get_df_of_phase_numbers_and_ints(mask=mask)
    dur_as_absolut_numbers = df.groupby(by=ColNames.PHASE_NUMBERS)[ColNames.INTS].transform("count")

    return Series(dur_as_absolut_numbers * np.sign(df[ColNames.PHASE_NUMBERS]))


@validate_and_process_mask
def get_mask_on_starts(mask: Series | DataFrame) -> Series:
    """
    Identify positions where True phases start in a boolean mask.

    This function detects the beginning of each consecutive True phase (on-period) in the input
    boolean mask. A True phase starts at:
      - Positions where the mask transitions from False to True, OR
      - The first position of the mask if it is True


    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).

    Returns:
        pandas.Series: A boolean Series with the same index and length as the input mask.
                       Contains True at positions where a True phase starts, False elsewhere.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column

    Examples:
        >>> mask = Series([True, True, False, False, True, False, True, True])
        >>> get_mask_on_starts(mask)
        0     True   # First position, mask starts with True
        1    False   # Continuation of True phase
        2    False   # False phase, no start
        3    False   # False phase continues
        4     True   # Transition from False to True (start of new True phase)
        5    False   # False phase (not a True start)
        6     True   # Transition from False to True (start of new True phase)
        7    False   # Continuation of True phase
        dtype: bool

        >>> mask = Series([False, False, True, False])
        >>> # No True at position 0, transition to True at position 2
        >>> get_mask_on_starts(mask)
        0    False
        1    False
        2     True   # Transition from False to True
        3    False
        dtype: bool

        >>> mask = Series([True, False, True, True])
        >>> # True at position 0 (special case: first element), then transition at position 2
        >>> get_mask_on_starts(mask)
        0     True   # First element is True
        1    False
        2     True   # Transition from False to True
        3    False   # Continuation of True phase
        dtype: bool

    Use cases:
      - Detecting the onset of events or anomalies in time series data
      - Marking the beginning of on-periods for analysis purposes
      - Segmenting data into consecutive True/False phases
      - Identifying boundaries between inactive and active periods

    Notes:
      - The first position receives special treatment: it is marked as True if mask.iloc[0] is True,
        regardless of any preceding value (since there is no previous value to compare).
      - This function is typically used in combination with `get_mask_on_stops` to identify both
        phase boundaries.
      - See also: `get_mask_on_stops` for detecting where True phases end, and
        `get_mask_phase_numbers` for assigning phase identifiers.
    """
    signal_of_ones_and_zeros = convert_mask_into_ones_and_zeros(mask=mask)
    signal_diff = _get_signal_diff(signal_of_ones_and_zeros=signal_of_ones_and_zeros)
    mask_starts = signal_diff == 1

    if not mask.empty and mask.iloc[0]:
        mask_starts = mask_starts.copy()
        mask_starts.iloc[0] = mask.iloc[0]

    return mask_starts


@validate_and_process_mask
def get_mask_on_stops(mask: Series | DataFrame) -> Series:
    """
    Identify the positions that mark the end of True phases in a boolean mask.

    This function returns a boolean Series (same index and length as `mask`) with True at
    positions that correspond to the last True element of each consecutive True run
    (i.e. where a True phase ends). The detection works by:
      1. Finding positions where an off-phase (False) starts (`_get_mask_off_starts`),
      2. Shifting those positions backward by one (shift(-1)) so the True value immediately
         before the off-start is marked (the end of the preceding True phase).
      3. Applying a small boundary rule for the final element:
         - If the input mask ends with True, the final index is explicitly marked True,
           because there is no subsequent False transition to detect an end.
         - Otherwise the final index is set to False.

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).

    Returns:
        pandas.Series: Boolean Series marking end positions of True phases. True at the last
                       index of each True-run; False elsewhere.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column

    Examples:
        >>> mask = Series([True, True, False, False, True, False, True, True])
        >>> get_mask_on_stops(mask)
        0    False
        1     True   # end of first True run (index 1)
        2    False
        3    False
        4     True   # single True at index 4 ends immediately
        5    False
        6    False
        7     True   # last True run ends at final index (mask[-1] is True)
        dtype: bool

        >>> mask = Series([False, False, True, True, False])
        >>> get_mask_on_stops(mask)
        0    False
        1    False
        2    False
        3     True   # end of True run at index 3
        4    False
        dtype: bool

        >>> mask = Series([False, False, False])
        >>> get_mask_on_stops(mask)
        0    False
        1    False
        2    False
        dtype: bool

        >>> mask = Series([True, True, True])
        >>> get_mask_on_stops(mask)
        0    False
        1    False
        2     True   # only the last element is marked because the run ends at series end
        dtype: bool

    Notes:
      - For masks that never switch from True to False (all True), the function will mark
        the final element True (end at the end of the series) and all previous elements False.
      - For masks that never contain True the result is all False.
    """
    mask_off_starts = get_mask_off_starts(mask=mask)
    mask_on_stops = mask_off_starts.shift(-1)

    if not mask.empty:
        mask_on_stops = mask_on_stops.copy()
        if mask.iloc[-1]:
            mask_on_stops.iloc[-1] = True
        else:
            mask_on_stops.iloc[-1] = False

    return mask_on_stops


@validate_and_process_mask
def get_mask_off_starts(mask: Series | DataFrame, exclude_first: bool = False) -> Series:
    """
    Identify positions where False phases start in a boolean mask.

    This function detects the beginning of each consecutive False phase (off-period) in the input
    boolean mask. A False phase starts at:
      - Positions where the mask transitions from True to False, OR
      - The first position of the mask if it is False (unless exclude_first=True)

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).
        exclude_first (bool, optional): If True, the first position will not be marked as a
            False phase start, even if mask.iloc[0] is False. Defaults to False.

    Returns:
        pandas.Series: A boolean Series with the same index and length as the input mask.
                       Contains True at positions where a False phase starts, False elsewhere.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column
         (raised by `@validate_and_process_mask`).

    Examples:
        >>> mask = Series([True, True, False, False, True, False, True, True])
        >>> get_mask_off_starts(mask)
        0    False
        1    False
        2     True   # Transition from True to False (start of off-phase)
        3    False   # Continuation of False phase
        4    False
        5     True   # Transition from True to False (start of off-phase)
        6    False
        7    False
        dtype: bool

        >>> mask = Series([False, False, True, True])
        >>> get_mask_off_starts(mask)
        0     True   # First position, mask starts with False
        1    False   # Continuation of False phase
        2    False
        3    False
        dtype: bool

        >>> mask = Series([False, True, False, False])
        >>> # With exclude_first=True, the first False is not marked as a start
        >>> get_mask_off_starts(mask, exclude_first=True)
        0    False
        1    False
        2     True   # Transition from True to False (start of off-phase)
        3    False
        dtype: bool

        >>> mask = Series([False, True, False, False])
        >>> # With exclude_first=False (default), the first False is marked as a start
        >>> get_mask_off_starts(mask, exclude_first=False)
        0     True   # First position is False
        1    False
        2     True   # Transition from True to False
        3    False
        dtype: bool

    Use cases:
      - Detecting the onset of inactivity or off-periods in time series data
      - Marking the beginning of off-periods for analysis purposes
      - Segmenting data into consecutive True/False phases
      - Identifying boundaries between active and inactive periods
      - Used by `get_mask_phase_numbers` with exclude_first=True to avoid double-counting phases
    """
    signal_of_ones_and_zeros = convert_mask_into_ones_and_zeros(mask=mask)
    signal_diff = _get_signal_diff(signal_of_ones_and_zeros=signal_of_ones_and_zeros)
    mask_off_starts = signal_diff == -1

    if not mask.empty and not mask.iloc[0] and not exclude_first:
        mask_off_starts = mask_off_starts.copy()
        mask_off_starts.iloc[0] = True

    return mask_off_starts


# No validation or processing since get_mask_on_stops is already validated and processed
def get_mask_off_stops(mask: Series | DataFrame) -> Series:
    """
    Identify the positions that mark the end of False phases in a boolean mask.

    This function returns a boolean Series (same index and length as `mask`) with True at
    positions that correspond to the last False element of each consecutive False run
    (i.e. where a False phase ends). It operates by inverting the input mask and delegating
    to `get_mask_on_stops`, effectively finding where on-phases end in the inverted mask,
    which corresponds to where off-phases end in the original mask.

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask indicating on/off (True/False).
            If a DataFrame is provided, it will be converted to a Series by the
            `@validate_and_process_mask` decorator via `get_mask_on_stops`.

    Returns:
        pandas.Series: Boolean Series marking end positions of False phases. True at the last
                       index of each False-run; False elsewhere. The index matches the input
                       mask's index and length.

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `get_mask_on_stops`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column
         (raised by `get_mask_on_stops`).

    Examples:
        >>> mask = Series([True, True, False, False, True, False, True, True])
        >>> get_mask_off_stops(mask)
        0    False
        1    False
        2    False
        3     True   # end of first False run (index 3)
        4    False
        5     True   # single False at index 5 ends immediately
        6    False
        7    False
        dtype: bool

        >>> mask = Series([False, False, True, True, False])
        >>> get_mask_off_stops(mask)
        0    False
        1     True   # end of False run at index 1
        2    False
        3    False
        4     True   # last False run ends at final index (mask[-1] is False)
        dtype: bool

        >>> mask = Series([True, True, True])
        >>> get_mask_off_stops(mask)
        0    False
        1    False
        2    False
        dtype: bool

        >>> mask = Series([False, False, False])
        >>> get_mask_off_stops(mask)
        0    False
        1    False
        2     True   # only the last element is marked because the run ends at series end
        dtype: bool

    Use cases:
      - Detecting the end of inactivity or off-periods in time series data
      - Marking the ending of off-periods for analysis purposes
      - Segmenting data to identify boundaries within False/off phases
      - Identifying timestamps or indices where inactivity periods conclude
      - Complementing `get_mask_on_stops` for symmetric on/off phase analysis

    """
    return get_mask_on_stops(mask=~mask)


@validate_and_process_mask
def convert_mask_into_ones_and_zeros(mask: Series | DataFrame) -> Series:
    """
    Convert a boolean mask into a numeric Series with integer values (0 and 1).

    This function performs a simple type conversion of boolean values to their numeric equivalents:
    True becomes 1 and False becomes 0. This conversion is useful for mathematical operations,
    aggregations, or when a numeric representation of boolean logic is required.

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask to convert to numeric values.
            If a DataFrame is provided, it will be converted to a Series by the
            `@validate_and_process_mask` decorator (via `convert_frame2series`).

    Returns:
        pandas.Series: A numeric Series with the same index and length as the input mask,
                       where True values are converted to 1 and False values are converted to 0.
                       The dtype of the returned Series is int64 (or equivalent platform int).

    Raises:
        TypeError: If `mask` does not have boolean dtype (raised by `@validate_and_process_mask`).
        DataframeDimensionError: If `mask` is a DataFrame with more than one column
         (raised by `@validate_and_process_mask`).

    Examples:
        >>> mask = Series([True, False, True, False, True])
        >>> convert_mask_into_ones_and_zeros(mask)
        0    1
        1    0
        2    1
        3    0
        4    1
        dtype: int64

        >>> # Empty mask returns empty Series
        >>> mask = Series([], dtype=bool)
        >>> convert_mask_into_ones_and_zeros(mask)
        Series([], dtype: int64)

    Use cases:
      - Preparing boolean masks for mathematical calculations (sum, mean, etc.)
      - Converting on/off indicator logic to numeric form for statistics
      - Enabling operations that require numeric types (e.g., multiplication, averaging)
      - Creating binary signals from boolean phase indicators
      - Computing statistics on phase presence (counting True occurrences)
    """
    return mask.astype(int)


# No validation or processing since convert_mask_into_signal_of_ones_and_zeros is already validated and processed
def convert_mask_into_ones_and_negative_ones(mask: Series | DataFrame) -> Series:
    """
    Convert a boolean mask into a signed numeric Series with values of 1 and -1.

    This function performs a type conversion of boolean values to their signed numeric equivalents:
    True becomes 1 and False becomes -1. This creates a symmetric representation centered around 0,
    useful for operations like cumulative sums with sign preservation, signal processing, or
    mathematical transformations that require symmetric encoding.

    Args:
        mask (pandas.Series or pandas.DataFrame): Boolean mask to convert to signed numeric values.
            If a DataFrame is provided, it will be converted to a Series by the
            `convert_frame2series` function used by the decorated caller.

    Returns:
        pandas.Series: A signed numeric Series with the same index and length as the input mask,
                       where True values are converted to 1 and False values are converted to -1.
                       The dtype of the returned Series is int64 (or equivalent platform int).

    Raises:
        TypeError: If the input mask is not a Series or DataFrame.
        DataframeDimensionError: If `mask` is a DataFrame with more than one column
         (raised by `@validate_and_process_mask`).

    Examples:
        >>> mask = Series([True, False, True, False, True])
        >>> convert_mask_into_ones_and_negative_ones(mask)
        0     1
        1    -1
        2     1
        3    -1
        4     1
        dtype: int64

    Use cases:
      - Creating signed signals for mathematical operations (e.g., phase detection)
      - Enabling cumulative sum operations that preserve direction/sign
      - Computing phase transitions via signal differences (True-to-False = -2, False-to-True = 2)
      - Multiplying with numeric data to flip sign based on phase (True vs False)
      - Statistical analysis requiring symmetric bipolar encoding
      - Feeding into groupby operations combined with cumulative sums
    """
    signal_of_ones_and_zeros = convert_mask_into_ones_and_zeros(mask=mask)

    return signal_of_ones_and_zeros * 2 - 1


def _get_start_numbers(mask_on_starts: Series) -> Series:
    return mask_on_starts.astype(int).cumsum()


def _get_stop_numbers(mask_off_starts: Series, first_mask_item: bool) -> Series:
    if first_mask_item:
        return mask_off_starts.astype(int).cumsum() * -1
    else:
        return mask_off_starts.astype(int).cumsum() * -1 - 1


def _get_df_of_phase_numbers_and_ints(mask: Series | DataFrame) -> DataFrame:
    phase_numbers = get_mask_phase_numbers(mask=mask)
    ints = convert_mask_into_ones_and_negative_ones(mask=mask)

    return DataFrame(data={ColNames.INTS: ints, ColNames.PHASE_NUMBERS: phase_numbers})


def _get_signal_diff(signal_of_ones_and_zeros: Series | DataFrame) -> Series:
    signal_of_ones_and_zeros = convert_frame2series(signal_of_ones_and_zeros)
    return signal_of_ones_and_zeros.diff()
