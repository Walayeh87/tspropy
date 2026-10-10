from pandas import DataFrame


def split_df_into_list_of_phases(df: DataFrame, phase_nr_col: str) -> list[DataFrame]:
    """Split a DataFrame into a list of DataFrames based on unique values in a specified column.

    Args:
        df (DataFrame): The input DataFrame to be split.
        phase_nr_col (str): The name of the column in the DataFrame that contains the phase numbers.

    Returns:
        list[DataFrame]: A list of DataFrames, each containing rows corresponding to a unique phase number.

    Raises:
        ValueError: If the input DataFrame has less than two columns or if the specified phase_nr_col does not exist
         in the DataFrame columns.
    """
    _validate_params_values(df=df, phase_nr_col=phase_nr_col)

    if len(df) == 0:
        return []

    return [group for _, group in df.groupby(phase_nr_col, sort=False)]


def _validate_params_values(df: DataFrame, phase_nr_col: str) -> None:
    if len(df.columns) < 2:
        raise ValueError("The input DataFrame must have at least two columns!")

    if phase_nr_col not in df.columns:
        raise ValueError(f"The specified phase_nr_col '{phase_nr_col}' does not exist in the DataFrame columns!")
