import pandas as pd

from src.data_manipulation.core.advanced.df_interpolator import interpolate_df
from src.data_manipulation.custom_objects.plotter_objects import Labels
from src.plotters.plotly_plotters import plot_datetime_data

df = pd.read_csv(
    r"../examples/data/dataset_with_missing_data.csv",
    index_col="datetime",
    parse_dates=True,
)
# loads_cols = ["Grid Load", "PV Generation"]
#
# df_part = df.loc["2017-12-28"]
# df_part = df_part.resample("15min").mean()
# df_part = df_part[loads_cols]
#
# df_part["non-numerical column"] = "a"
#
# df_part = df_part[["Grid Load", "non-numerical column", "PV Generation"]]
#
# df_part.loc["2017-12-28 09:15":"2017-12-28 12:00", "non-numerical column"] = np.nan
#
# df_part.loc["2017-12-28 07:15":"2017-12-28 09:00", "PV Generation"] = np.nan
# df_part.loc["2017-12-28 12:00":"2017-12-28 12:30", "PV Generation"] = np.nan
#
# df_part.loc["2017-12-28 14:00":"2017-12-28 14:15", "Grid Load"] = np.nan
#
# df_part = df_part.round(2)
# df_part.to_csv("../examples/data/dataset_with_missing_data.csv", index_label="datetime")

fig = plot_datetime_data(
    df=df,
    labels=Labels(y1_label="Load in MW"),
)
# fig.show()

interpolation_result = interpolate_df(
    df=df,
    # interpolation_limits_mapper={
    #     "Grid Load": PhaseDuration("30min"),
    #     "PV Generation": PhaseDuration("1h"),
    # },
)
