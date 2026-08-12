import pandas as pd

from src.data_manipulation.core.advanced.df_interpolator import interpolate_df
from src.data_manipulation.custom_objects.phase_duration import PhaseDuration
from src.data_manipulation.custom_objects.plotter_objects import Labels
from src.plotters.plotly_plotters import plot_datetime_data

df = pd.read_csv(
    r"../examples/data/dataset_with_missing_data.csv",
    index_col="datetime",
    parse_dates=True,
)

# The dataset consists of 3 columns; 2 numerical ones "Grid Load" and "PV Generation",
# and 1 non-numerical "non-numerical column".
# The following plot illustrates the numerical columns that show missing data.
print(df.head())

fig1 = plot_datetime_data(df=df, labels=Labels(y1_label="Load in MW"), title="Original Dataset")
fig1.show()

# pd.interpolate vs tspropy.interpolate_df
# AI should fill in this as a table. The InterpolationResult object should be explained, too.

# tspropy.interpolate_df use cases
# 1- Simple case: interpolate all numerical columns without limits with "time" interpolation method.
# Since "non-numerical column" is not interpolatable,it will be preserved with the same column order as the input. It
# will not show up on the plots.
interpolation_result1 = interpolate_df(df=df)
fig2 = plot_datetime_data(
    df=interpolation_result1.interpolated_df,
    labels=Labels(y1_label="Load in MW"),
    title="No limit & time method",
)
fig2.show()

# Comparison between the dataset before and after the interpolation
print(df.loc["2017-12-28 13:30":"2017-12-28 15:00"])
print(interpolation_result1.interpolated_df.loc["2017-12-28 13:30":"2017-12-28 15:00"])

# interpolation_result1.interpolation_mask marks with Trues the interpolated cells, while marks the rest with Falses.
print(interpolation_result1.interpolation_mask.loc["2017-12-28 13:30":"2017-12-28 15:00"])

# interpolation_result1.nan_statistics gives an informative overview about the nan count in the dataset before and after
# the interpolation
print(interpolation_result1.nan_statistics)

# 2- Advanced cases
# A- Specify an interpolation limit of "1h" for "PV Generation" and no limit for "Grid Load". This will interpolate
# only the second gap (the short gap) in "PV Generation" and skipping the first gap (about 2h long) unchanged. The gap
# in "Grid Load" will be interpolated since there is no limit.
interpolation_result2 = interpolate_df(
    df=df,
    limits_mapper={
        "PV Generation": PhaseDuration("1h"),
        # No limit for "Grid Load"
    },
)

fig3 = plot_datetime_data(
    df=interpolation_result2.interpolated_df,
    labels=Labels(y1_label="Load in MW"),
    title="1h limit for PV Generation & time method for both",
)
fig3.show()

# Check the nan statistics
print(interpolation_result2.nan_statistics)

# B- Specify different interpolation methods for each column
interpolation_result3 = interpolate_df(
    df=df,
    methods_mapper={
        "PV Generation": "linear",
        "Grid Load": "cubic",
    },
)

fig4 = plot_datetime_data(
    df=interpolation_result3.interpolated_df,
    labels=Labels(y1_label="Load in MW"),
    title="No limit & different methods",
)
fig4.show()
