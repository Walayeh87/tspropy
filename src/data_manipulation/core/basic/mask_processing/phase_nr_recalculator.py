from pandas import Series


def recalculate_phase_nr(phase_nr_series: Series) -> Series:
    """
    Recalculate the phase numbers in a pandas Series to ensure they are sequentially numbered.

    This function takes a pandas Series of phase numbers, validates its type, data type, and monotonicity,
    and then recalculates the phase numbers such that each unique phase is assigned a sequential number.

    Args:
        phase_nr_series (Series): A pandas Series containing the phase numbers. Must be monotonic increasing,
                           and consist of integers or floats.

    Returns:
        Series: A pandas Series with recalculated sequential phase numbers.

    Raises:
        TypeError: If `phase_nr_series` contains non-integer or non-float values.
        ValueError: If `phase_nr_series` is not monotonic increasing.
    """
    if phase_nr_series.empty:
        return phase_nr_series

    if phase_nr_series.dtype not in [int, float]:
        raise TypeError(f"phase_nr_series must consist of only integer or float. {phase_nr_series.dtype} was passed!")

    if len(phase_nr_series) == 1:
        return Series([1], index=phase_nr_series.index)

    if not phase_nr_series.is_monotonic_increasing:
        raise ValueError("phase_nr_series must be a monotonic increasing Series!")

    new_phase_nr = phase_nr_series.diff()
    new_phase_nr.loc[new_phase_nr > 0] = 1
    new_phase_nr.iloc[0] = 1

    return new_phase_nr.cumsum()


if __name__ == "__main__":
    test_series = Series([0, 0, 2, 2, 4, 5, 5])
    print(recalculate_phase_nr(phase_nr_series=test_series))
