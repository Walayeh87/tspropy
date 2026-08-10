import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype

from src.data_manipulation.core.basic.mask_processing.mask_properties import get_mask_phase_durations
from src.data_manipulation.custom_objects.phase_duration import PhaseDuration
from src.data_manipulation.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ColStatistics:
    filled: int
    remaining: int


@dataclass
class InterpolationResult:
    interpolated_data: DataFrame | Series
    interpolation_mask: DataFrame | Series
    interpolation_statistics: dict[str, ColStatistics]


InterpolationMethod = Literal[
    "time",
    "linear",
    "index",
    "values",
    "nearest",
    "zero",
    "slinear",
    "quadratic",
    "cubic",
    "barycentric",
    "polynomial",
    "krogh",
    "piecewise_polynomial",
    "spline",
    "pchip",
    "akima",
    "cubicspline",
    "from_derivatives",
]


def interpolate_df(
    df: DataFrame,
    interpolation_limits_mapper: dict[str, PhaseDuration] | None = None,
    default_interpolation_limit: PhaseDuration | None = None,
    interpolation_methods_mapper: dict[str, InterpolationMethod] | None = None,
    default_interpolation_method: InterpolationMethod = "time",
) -> InterpolationResult:
    """
    Interpolate missing values (NaN) in numeric columns of a DataFrame.

    The function:
    - Validates any provided per-column mappings (limits and methods) against the
      DataFrame and ensures they target interpolatable (numeric) columns.
    - Builds a mapping of interpolation methods and limits for all columns, using
      defaults where specific mappings are not supplied.
    - Interpolates each numeric column using pandas.Series.interpolate with the
      chosen method and respects "longest gap to fill" limits by reverting
      interpolations for gaps longer than the provided PhaseDuration.
    - Preserves non-numeric columns and returns a combined DataFrame with the
      same column order as the input.

    Args:
        df: Input pandas DataFrame. The function copies the DataFrame internally
            and does not mutate the original.
        interpolation_limits_mapper: Optional dict mapping column names -> PhaseDuration.
            For each specified column, gaps longer than the provided PhaseDuration
            will not be filled (they remain NaN).
        default_interpolation_limit: Default PhaseDuration applied to columns that
            are not listed in `interpolation_limits_mapper`. If None, no duration
            restriction is applied for those columns.
        interpolation_methods_mapper: Optional dict mapping column names -> interpolation method
            (one of the literal values defined in `InterpolationMethod`). Columns not listed
            will use `default_interpolation_method`.
        default_interpolation_method: Interpolation method to use for columns not
            present in `interpolation_methods_mapper`. Defaults to "time".

    Returns:
        InterpolationResult: includes:
          - interpolated_data (DataFrame): DataFrame with numeric columns interpolated.
          - interpolation_mask (DataFrame of booleans): True where the original value
              was NaN and now is non-NaN (i.e., values that were interpolated).
          - interpolation_statistics (dict[str, ColStatistics]): per-column statistics
              with counts for filled and remaining NaNs.

    Raises:
        ValueError: If a column name provided in `interpolation_limits_mapper` or
            `interpolation_methods_mapper` does not exist in `df` or is not numeric
            (i.e., not interpolatable).

    Notes:
    - Only numeric dtypes (as determined by pandas.api.types.is_numeric_dtype) are candidates.
    - If the DataFrame is empty, contains no numeric columns, or contains no NaNs,
      the returned result will contain the original DataFrame and statistics reflect
      zero or the unchanged gap counts.
    """
    if isinstance(interpolation_limits_mapper, dict):
        for col_name in interpolation_limits_mapper.keys():
            if col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' is not a column of the entered df. Valid columns are: {list(df)}"
                )

    if isinstance(interpolation_methods_mapper, dict):
        for col_name in interpolation_methods_mapper.keys():
            if col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' is not a column of the entered df. Valid columns are: {list(df)}"
                )

    interpolatable_columns = _get_interpolatable_columns(df=df)

    if isinstance(interpolation_limits_mapper, dict):
        for column in interpolation_limits_mapper.keys():
            if column not in interpolatable_columns:
                raise ValueError(
                    f"Column '{column}' is not interpolatable. Valid interpolatable columns are:"
                    f" {list(interpolatable_columns)}"
                )

    if isinstance(interpolation_methods_mapper, dict):
        for column in interpolation_methods_mapper.keys():
            if column not in interpolatable_columns:
                raise ValueError(
                    f"Column '{column}' is not interpolatable. Valid interpolatable columns are:"
                    f" {list(interpolatable_columns)}"
                )

    df = df.copy()

    interpolation_mask = _create_interpolation_mask(df=df, interpolated_df=df)
    if df.shape == (0, 0):
        logger.info(
            "No interpolation can be performed since the passed data is empty! The data passed is returned unchanged!"
        )
        interpolation_statistics = _create_interpolation_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_data=df,
            interpolation_mask=interpolation_mask,
            interpolation_statistics=interpolation_statistics,
        )

    if len(interpolatable_columns) == 0:
        logger.warning("No interpolatable columns found. The data passed is returned unchanged!")
        interpolation_statistics = _create_interpolation_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_data=df,
            interpolation_mask=interpolation_mask,
            interpolation_statistics=interpolation_statistics,
        )

    if df.isna().sum().sum() == 0:
        logger.info("The entered df has no gaps. The data passed is returned unchanged!")
        interpolation_statistics = _create_interpolation_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_data=df,
            interpolation_mask=interpolation_mask,
            interpolation_statistics=interpolation_statistics,
        )

    methods2use = _get_methods2use(
        columns=list(df),
        interpolation_methods_mapper=interpolation_methods_mapper,
        default_interpolation_method=default_interpolation_method,
    )

    limits2use = _get_limits2use(
        columns=list(df),
        interpolation_limits_mapper=interpolation_limits_mapper,
        default_interpolation_limit=default_interpolation_limit,
    )

    interpolated_data_as_dict = {
        column: interpolate_series(series=df[column], method=methods2use[column], longest_gap2fill=limits2use[column])
        for column in interpolatable_columns
    }

    df_of_interpolatable_cols = DataFrame(interpolated_data_as_dict, index=df.index)

    cols2exclude = df.columns.difference(df_of_interpolatable_cols.columns)

    interpolated_df = pd.concat([df_of_interpolatable_cols, df[cols2exclude]], axis=1)

    interpolated_df = _resort_df_cols(interpolated_df=interpolated_df, original_cols=list(df))

    interpolation_mask = _create_interpolation_mask(df=df, interpolated_df=interpolated_df)

    interpolation_statistics = _create_interpolation_statistics(df=df, interpolated_df=interpolated_df)

    return InterpolationResult(
        interpolated_data=interpolated_df,
        interpolation_mask=interpolation_mask,
        interpolation_statistics=interpolation_statistics,
    )


def _create_interpolation_mask(df: DataFrame, interpolated_df: DataFrame) -> DataFrame:
    return df.isna() & interpolated_df.notna()


def _create_interpolation_statistics(df: DataFrame, interpolated_df: DataFrame) -> dict[str, ColStatistics]:
    interpolation_statistics = {}
    for col in df.columns:
        original_gaps = df[col].isna().sum().sum()
        remaining_gaps = interpolated_df[col].isna().sum().sum()
        interpolated_gaps = original_gaps - remaining_gaps

        col_statistics = ColStatistics(filled=interpolated_gaps, remaining=remaining_gaps)
        interpolation_statistics[col] = col_statistics

    return interpolation_statistics


def _resort_df_cols(interpolated_df: DataFrame, original_cols: list) -> DataFrame:
    return interpolated_df[original_cols]


def interpolate_series(series: Series, method: InterpolationMethod, longest_gap2fill: PhaseDuration | None) -> Series:
    if longest_gap2fill is None:
        return series.interpolate(method=method, limit_direction="forward", limit_area="inside")

    interpolated_series = series.interpolate(method=method, limit_direction="forward", limit_area="inside")
    interpolated_series = _override_originally_long_gaps_with_nans(
        series=series, interpolated_series=interpolated_series, longest_gap2fill=longest_gap2fill
    )

    return interpolated_series


def _override_originally_long_gaps_with_nans(
    series: Series, interpolated_series: Series, longest_gap2fill: PhaseDuration
) -> Series:
    gap_dur = get_mask_phase_durations(mask=series.isna())
    interpolated_series[gap_dur > longest_gap2fill.value] = np.nan

    return interpolated_series


def _get_limits2use(
    columns: list,
    interpolation_limits_mapper: dict[str, PhaseDuration] | None,
    default_interpolation_limit: PhaseDuration | None,
) -> dict:
    limits2use = {column: default_interpolation_limit for column in columns}
    if interpolation_limits_mapper is not None:
        limits2use.update(interpolation_limits_mapper)

    return limits2use


def _get_methods2use(
    columns: list,
    interpolation_methods_mapper: dict[str, InterpolationMethod] | None,
    default_interpolation_method: InterpolationMethod,
) -> dict:
    methods2use = {column: default_interpolation_method for column in columns}
    if interpolation_methods_mapper is not None:
        methods2use.update(interpolation_methods_mapper)

    return methods2use


def _get_interpolatable_columns(df: DataFrame) -> list:
    interpolatable_cols = []
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            interpolatable_cols.append(column)

    return interpolatable_cols


if __name__ == "__main__":
    # Example usage
    date_range = pd.date_range(start="2023-01-01", periods=10, freq="min")
    data = {
        "a": [1, np.nan, np.nan, 4, 5, np.nan, 7, 8, np.nan, 10],
        "b": [np.nan, 2, 3, np.nan, 6, 7, np.nan, 9, 10, 11],
        "e": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
        "c": ["bla", "blu", "blg", "blh", "bli", "blj", "blk", "bll", "blm", "bln"],
        "d": ["bla", "blu", "blg", "blh", "bli", "blj", "blk", "bll", "blm", "bln"],
    }
    df = DataFrame(data=data, index=date_range)

    interpolation_result = interpolate_df(
        df=DataFrame(),
        interpolation_limits_mapper=None,
        default_interpolation_limit=PhaseDuration("1min"),
        interpolation_methods_mapper=None,
        default_interpolation_method="time",
    )
