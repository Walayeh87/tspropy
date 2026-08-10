from pandas import DataFrame, Series

from src.data_manipulation.utils.custom_errors import DataframeDimensionError


def convert_series2frame(data: Series | DataFrame, name: str | None = None) -> DataFrame:
    """
    Converts a pandas Series or DataFrame into a DataFrame.

    If the input is already a DataFrame, it is returned as-is. If the input is a Series, it is converted
    into a DataFrame. Optionally, a column name can be specified for the resulting DataFrame.

    Args:
        data (Series | DataFrame): The input data to be converted. Must be either a pandas Series or DataFrame.
        name (str, optional): The name to assign to the column if the input is a Series. Defaults to None.

    Returns:
        DataFrame: The resulting DataFrame after conversion.
    """
    if isinstance(data, DataFrame):
        return data

    if isinstance(name, str) and len(name) == 0:
        raise ValueError("The 'name' parameter cannot be an empty string.")

    return data.to_frame() if name is None else data.to_frame(name=name)


def convert_frame2series(data: Series | DataFrame) -> Series:
    """
    Convert a pandas DataFrame (with a single column) into a pandas Series.

    This function accepts either a pandas Series or DataFrame and returns a pandas Series.
    - If `data` is already a Series it is returned unchanged.
    - If `data` is an empty DataFrame with shape (0, 0), an empty Series is returned.
    - If `data` is a DataFrame with exactly one column, that column is returned as a Series.
    - If `data` is a DataFrame with multiple columns, a DataframeDimensionError is raised.

    Args:
        data (Series | DataFrame): Input data to convert. Must be either a pandas Series
            or a pandas DataFrame.

    Returns:
        pd.Series: The resulting pandas Series. The returned Series will preserve the index
            of the input where applicable.

    Raises:
        DataframeDimensionError: If `data` is a DataFrame with more than one column.

    Notes:
        - For a DataFrame with shape (0, 1) (zero rows, one column) the single column is
          returned as an empty Series. Only the special case (0, 0) is normalized to an
          entirely empty Series instance.
        - The function uses `data.iloc[:, 0]` to extract the first (and only) column from
          a 1-column DataFrame.
    """
    if isinstance(data, Series):
        return data

    if data.shape == (0, 0):
        return Series()

    is_df_with_multiple_cols = data.shape[1] != 1
    if is_df_with_multiple_cols:
        raise DataframeDimensionError("'data' must be a pd.DataFrame with only one column!")

    return data.iloc[:, 0]
