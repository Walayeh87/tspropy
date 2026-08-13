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
class NanStatistics:
    original: int
    filled: int
    remaining: int


@dataclass
class InterpolationResult:
    interpolated_df: DataFrame
    interpolation_mask: DataFrame
    nan_statistics: dict[str, NanStatistics]


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
    limits_mapper: dict[str, PhaseDuration] | None = None,
    default_limit: PhaseDuration | None = None,
    methods_mapper: dict[str, InterpolationMethod] | None = None,
    default_method: InterpolationMethod = "time",
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
        limits_mapper: Optional dict mapping column names -> PhaseDuration.
            For each specified column, gaps longer than the provided PhaseDuration
            will not be filled (they remain NaN).
        default_limit: Default PhaseDuration applied to columns that
            are not listed in `limits_mapper`. If None, no duration
            restriction is applied for those columns.
        methods_mapper: Optional dict mapping column names -> interpolation method
            (one of the literal values defined in `InterpolationMethod`). Columns not listed
            will use `default_method`.
        default_method: Interpolation method to use for columns not
            present in `methods_mapper`. Defaults to "time".

    Returns:
        InterpolationResult: includes:
          - interpolated_df (DataFrame): DataFrame with numeric columns interpolated.
          - interpolation_mask (DataFrame of booleans): True where the original value
              was NaN and now is non-NaN (i.e., values that were interpolated).
          - nan_statistics (dict[str, NanStatistics]): per-column statistics
              with counts for filled and remaining NaNs.

    Raises:
        ValueError: If a column name provided in `limits_mapper` or
            `methods_mapper` does not exist in `df` or is not numeric
            (i.e., not interpolatable).

    Notes:
    - Only numeric dtypes (as determined by pandas.api.types.is_numeric_dtype) are candidates.
    - If the DataFrame is empty, contains no numeric columns, or contains no NaNs,
      the returned result will contain the original DataFrame and statistics reflect
      zero or the unchanged gap counts.
    """
    if isinstance(limits_mapper, dict):
        for col_name in limits_mapper.keys():
            if col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' is not a column of the entered df. Valid columns are: {list(df)}"
                )

    if isinstance(methods_mapper, dict):
        for col_name in methods_mapper.keys():
            if col_name not in df.columns:
                raise ValueError(
                    f"Column '{col_name}' is not a column of the entered df. Valid columns are: {list(df)}"
                )

    interpolatable_columns = _get_interpolatable_columns(df=df)

    if isinstance(limits_mapper, dict):
        for column in limits_mapper.keys():
            if column not in interpolatable_columns:
                raise ValueError(
                    f"Column '{column}' is not interpolatable. Valid interpolatable columns are:"
                    f" {list(interpolatable_columns)}"
                )

    if isinstance(methods_mapper, dict):
        for column in methods_mapper.keys():
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
        nan_statistics = _create_nan_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_df=df,
            interpolation_mask=interpolation_mask,
            nan_statistics=nan_statistics,
        )

    if len(interpolatable_columns) == 0:
        logger.warning("No interpolatable columns found. The data passed is returned unchanged!")
        nan_statistics = _create_nan_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_df=df,
            interpolation_mask=interpolation_mask,
            nan_statistics=nan_statistics,
        )

    if df.isna().sum().sum() == 0:
        logger.info("The entered df has no gaps. The data passed is returned unchanged!")
        nan_statistics = _create_nan_statistics(df=df, interpolated_df=df)

        return InterpolationResult(
            interpolated_df=df,
            interpolation_mask=interpolation_mask,
            nan_statistics=nan_statistics,
        )

    methods2use = _get_methods2use(
        columns=list(df),
        methods_mapper=methods_mapper,
        default_method=default_method,
    )

    limits2use = _get_limits2use(
        columns=list(df),
        limits_mapper=limits_mapper,
        default_limit=default_limit,
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

    nan_statistics = _create_nan_statistics(df=df, interpolated_df=interpolated_df)

    return InterpolationResult(
        interpolated_df=interpolated_df,
        interpolation_mask=interpolation_mask,
        nan_statistics=nan_statistics,
    )


def _create_interpolation_mask(df: DataFrame, interpolated_df: DataFrame) -> DataFrame:
    return df.isna() & interpolated_df.notna()


def _create_nan_statistics(df: DataFrame, interpolated_df: DataFrame) -> dict[str, NanStatistics]:
    nan_statistics = {}
    for col in df.columns:
        original_gaps = int(df[col].isna().sum().sum())
        remaining_gaps = int(interpolated_df[col].isna().sum().sum())
        interpolated_gaps = original_gaps - remaining_gaps

        col_statistics = NanStatistics(filled=interpolated_gaps, remaining=remaining_gaps, original=original_gaps)
        nan_statistics[col] = col_statistics

    return nan_statistics


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
    limits_mapper: dict[str, PhaseDuration] | None,
    default_limit: PhaseDuration | None,
) -> dict:
    limits2use = {column: default_limit for column in columns}
    if limits_mapper is not None:
        limits2use.update(limits_mapper)

    return limits2use


def _get_methods2use(
    columns: list,
    methods_mapper: dict[str, InterpolationMethod] | None,
    default_method: InterpolationMethod,
) -> dict:
    methods2use = {column: default_method for column in columns}
    if methods_mapper is not None:
        methods2use.update(methods_mapper)

    return methods2use


def _get_interpolatable_columns(df: DataFrame) -> list:
    interpolatable_cols = []
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            interpolatable_cols.append(column)

    return interpolatable_cols
