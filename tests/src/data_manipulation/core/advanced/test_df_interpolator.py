import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from src.data_manipulation.core.advanced.df_interpolator import (
    InterpolationMethod,
    InterpolationResult,
    NanStatistics,
    interpolate_df,
)
from src.data_manipulation.custom_objects.phase_duration import PhaseDuration


def create_df(column_mapper: dict, start: str = "2023-01-01", end: str = "2023-01-05", freq: str = "1D") -> DataFrame:
    idx = pd.date_range(start=start, end=end, freq=freq)

    return DataFrame(column_mapper, index=idx)


def create_interpolation_mask(
    column_mapper: dict, start: str = "2023-01-01", end: str = "2023-01-05", freq: str = "1D"
) -> DataFrame:
    idx = pd.date_range(start=start, end=end, freq=freq)

    return DataFrame(column_mapper, index=idx)


df_with_no_gaps = create_df({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
df_with_no_interpolatable_cols = create_df({"a": ["x", "y", "z", "w", "v"], "b": ["p", "q", "r", "s", "t"]})
df_with_gaps = create_df({"a": [1, np.nan, 3, 4, 5], "b": [6, np.nan, np.nan, 9, 10]})
df_with_long_gaps = create_df({"a": [1, np.nan, np.nan, 4, 5], "b": [6, np.nan, np.nan, 9, 10]})
df_with_numerical_and_non_numerical_cols = create_df(
    {"numerical": [1, np.nan, 3, 4, 5], "non_numerical": ["x", "y", "z", "w", "v"]}
)


@pytest.mark.parametrize(
    "df, limits_mapper, default_limit, methods_mapper, default_method, expected_interpolation_result",
    [
        (
            df_with_no_gaps,
            {"not_existing_col": PhaseDuration("1D"), "b": PhaseDuration("1D")},
            "1min",
            None,
            "time",
            InterpolationResult(
                interpolated_df=DataFrame(),
                interpolation_mask=DataFrame(),
                nan_statistics={},
            ),
        ),
        (
            df_with_no_gaps,
            {"a": PhaseDuration("1D"), "b": PhaseDuration("1D")},
            "1min",
            {"not_existing_col": "linear"},
            "time",
            InterpolationResult(
                interpolated_df=DataFrame(),
                interpolation_mask=DataFrame(),
                nan_statistics={},
            ),
        ),
        (
            df_with_numerical_and_non_numerical_cols,
            {"numerical": PhaseDuration("1D"), "non_numerical": PhaseDuration("1D")},
            None,
            None,
            "time",
            InterpolationResult(
                interpolated_df=DataFrame(),
                interpolation_mask=DataFrame(),
                nan_statistics={},
            ),
        ),
        (
            df_with_numerical_and_non_numerical_cols,
            None,
            None,
            {"numerical": "time", "non_numerical": "time"},
            "time",
            InterpolationResult(
                interpolated_df=DataFrame(),
                interpolation_mask=DataFrame(),
                nan_statistics={},
            ),
        ),
    ],
)
def test_interpolate_df_with_invalid_column_names(
    df: DataFrame,
    limits_mapper: dict[str, PhaseDuration] | None,
    default_limit: PhaseDuration | None,
    methods_mapper: dict[str, InterpolationMethod] | None,
    default_method: InterpolationMethod,
    expected_interpolation_result: InterpolationResult,
) -> None:
    with pytest.raises(ValueError):
        interpolate_df(
            df=df,
            limits_mapper=limits_mapper,
            default_limit=default_limit,
            methods_mapper=methods_mapper,
            default_method=default_method,
        )


@pytest.mark.parametrize(
    "df, limits_mapper, default_limit, methods_mapper, default_method, expected_interpolation_result",
    [
        (
            DataFrame(),
            None,
            "1min",
            None,
            "time",
            InterpolationResult(
                interpolated_df=DataFrame(),
                interpolation_mask=DataFrame(),
                nan_statistics={},
            ),
        ),
        (
            df_with_no_gaps,
            None,
            PhaseDuration("1D"),
            None,
            "time",
            InterpolationResult(
                interpolated_df=df_with_no_gaps,
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, False, False, False, False], "b": [False, False, False, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=0, filled=0, remaining=0),
                    "b": NanStatistics(original=0, filled=0, remaining=0),
                },
            ),
        ),
        (
            df_with_no_interpolatable_cols,
            None,
            PhaseDuration("1D"),
            None,
            "time",
            InterpolationResult(
                interpolated_df=df_with_no_interpolatable_cols,
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, False, False, False, False], "b": [False, False, False, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=0, filled=0, remaining=0),
                    "b": NanStatistics(original=0, filled=0, remaining=0),
                },
            ),
        ),
        # Short gaps will be interpolated, the long will be kept
        (
            df_with_gaps,
            None,
            PhaseDuration("1D"),
            None,
            "time",
            InterpolationResult(
                interpolated_df=create_df({"a": [1, 2, 3, 4, 5], "b": [6, np.nan, np.nan, 9, 10]}),
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, True, False, False, False], "b": [False, False, False, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=1, filled=1, remaining=0),
                    "b": NanStatistics(original=2, filled=0, remaining=2),
                },
            ),
        ),
        # Short and long gaps will be interpolated
        (
            df_with_gaps,
            {"a": PhaseDuration("1D"), "b": PhaseDuration("2D")},
            None,
            None,
            "time",
            InterpolationResult(
                interpolated_df=create_df({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}),
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, True, False, False, False], "b": [False, True, True, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=1, filled=1, remaining=0),
                    "b": NanStatistics(original=2, filled=2, remaining=0),
                },
            ),
        ),
        # Short and long gaps will be interpolated (by using very long limits)
        (
            df_with_gaps,
            {"a": PhaseDuration("5D"), "b": PhaseDuration("20D")},
            None,
            None,
            "time",
            InterpolationResult(
                interpolated_df=create_df({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}),
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, True, False, False, False], "b": [False, True, True, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=1, filled=1, remaining=0),
                    "b": NanStatistics(original=2, filled=2, remaining=0),
                },
            ),
        ),
        # Limits are shorter than gaps -> no interpolation
        (
            df_with_long_gaps,
            {"a": PhaseDuration("1D"), "b": PhaseDuration("1D")},
            None,
            None,
            "time",
            InterpolationResult(
                interpolated_df=df_with_long_gaps,
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, False, False, False, False], "b": [False, False, False, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=2, filled=0, remaining=2),
                    "b": NanStatistics(original=2, filled=0, remaining=2),
                },
            ),
        ),
        # Interpolate all gaps by not specifying limits
        (
            df_with_long_gaps,
            None,
            None,
            None,
            "time",
            InterpolationResult(
                interpolated_df=df_with_no_gaps,
                interpolation_mask=create_interpolation_mask(
                    column_mapper={"a": [False, True, True, False, False], "b": [False, True, True, False, False]}
                ),
                nan_statistics={
                    "a": NanStatistics(original=2, filled=2, remaining=0),
                    "b": NanStatistics(original=2, filled=2, remaining=0),
                },
            ),
        ),
    ],
)
def test_interpolate_df(
    df: DataFrame,
    limits_mapper: dict[str, PhaseDuration] | None,
    default_limit: PhaseDuration | None,
    methods_mapper: dict[str, InterpolationMethod] | None,
    default_method: InterpolationMethod,
    expected_interpolation_result: InterpolationResult,
) -> None:
    interpolation_result = interpolate_df(
        df=df,
        limits_mapper=limits_mapper,
        default_limit=default_limit,
        methods_mapper=methods_mapper,
        default_method=default_method,
    )

    pd.testing.assert_frame_equal(
        interpolation_result.interpolated_df,
        expected_interpolation_result.interpolated_df,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        interpolation_result.interpolation_mask,
        expected_interpolation_result.interpolation_mask,
        check_dtype=False,
    )
