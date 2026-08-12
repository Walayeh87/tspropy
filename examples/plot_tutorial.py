import pandas as pd

from src.data_manipulation.custom_objects.location import Location
from src.data_manipulation.custom_objects.plotter_objects import (
    Axes,
    BuiltinRange,
    Color,
    ColorMapper,
    CustomRange,
    FontSizes,
    Labels,
    LineStyles,
    LineWidths,
    MaskRange,
    RangeConfig,
    RangePropertyName,
    StyleName,
    VLineConfig,
    VLinePropertyName,
    YLimits,
)
from src.plotters.plotly_plotters import plot_datetime_comparison, plot_datetime_data

# plot_datetime_data
# Basic plotting features
# Plotting all dataset columns on one axes. Legends, x-label "Time" and grid are shown in default
df = pd.read_csv(
    r"..\examples\data\load_and_price.csv",
    index_col="datetime",
    parse_dates=True,
)

loads_cols = ["Total Site Load", "Grid Load", "PV Generation"]
loads_df = df[loads_cols]

fig1 = plot_datetime_data(df=loads_df)
# fig1.show()

# Specifying y-limits and colors
fig2 = plot_datetime_data(
    df=loads_df,
    y_limits=YLimits(y1_limit=[-1, 15]),
    color_mapper=[
        ColorMapper(curve_name="Grid Load", color=Color.BLUE),
        ColorMapper(curve_name="PV Generation", color=Color.ORANGE),
        # "Total Site Load" is not specified -> a random color will be selected.
    ],
)
# fig2.show()

# Specifying curves to plot on different axes, a title and labels
fig3 = plot_datetime_data(
    df=df,
    # By default, curves on y1 have a solid line-style, while curves on y2 will have dashed line-style
    # (if y3 were specified, the curves on it would have a dashed-dotted line-style)
    axes=Axes(
        y1=loads_cols,
        y2="Day-Ahead Price",  # ["Day-Ahead Price"] works, too.
    ),
    title="Load and Price Data",
    labels=Labels(
        y1_label="Load in MW",
        y2_label="Price in €/MW",
    ),
)
# fig3.show()

# Advanced plotting features
# 1- Highlighted ranges
# A- Builtin ranges
# The total site load on 30th, 31st of December 2017 and the 1st of January 2018 is noticeably lower than in the other days.
# Highlighting the weekend and the public holiday explaines that these days are weekends and a publlic holidays, in which
# the energy consumption is usually lower than in working days.
fig4 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    range_config=[
        RangeConfig(
            range_type=BuiltinRange.WEEKENDS,  # Default color is gray
        ),
        RangeConfig(
            range_type=BuiltinRange.HOLIDAYS,
            property={RangePropertyName.COLOR: Color.CYAN},
            # Location is a mandatory input when range_type is BuiltinRange.HOLIDAYS
            location=Location(country_code="DE"),
        ),
    ],
)
# fig4.show()

# B- Custom ranges
# If a specific range needs to be highlighted to illustrate its importance, RangeConfig can be used
fig5 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    range_config=RangeConfig(
        range_type=CustomRange(
            range=["2017-12-30 12:00", "2017-12-30 18:00"],  # Random time range is used for this example
        ),
        property={RangePropertyName.COLOR: Color.CORAL},
    ),
)
# fig5.show()

# C- Mask ranges
# If a specific pattern in the data needs to be highlighted, a boolean mask can be used to define the range.
# In this example, the pv-surplus phases will be highlighted in orange.
mask_of_pv_surplus_phases = loads_df["Grid Load"] == 0

fig6 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    range_config=RangeConfig(
        range_type=MaskRange(range=mask_of_pv_surplus_phases),
        property={RangePropertyName.COLOR: Color.ORANGE},
    ),
)
# fig6.show()

# A combination of different rang types can be used on one plot
fig7 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    range_config=[
        RangeConfig(
            range_type=MaskRange(range=mask_of_pv_surplus_phases),
            property={RangePropertyName.COLOR: Color.ORANGE},
        ),
        RangeConfig(
            range_type=BuiltinRange.WEEKENDS,  # Default color is gray
        ),
    ],
)
# fig7.show()

# 2- Vertical lines
# It is used to emphasize specific points in time, or simply to split the plot using a uniform intervals.
# In this example, the plot will be split by vertical lines for each 12 hours. (grid is disabled for a better illustration)
fig8 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    v_line_config=VLineConfig(
        position=loads_df.index.floor("12h").unique(),
        property={
            VLinePropertyName.COLOR: Color.BLACK,
            VLinePropertyName.WIDTH: 10,
        },
    ),
    show_grid=False,
)
# fig8.show()

# Secondary features
# plot_datetime_data offers many of secondary features for customizing the appearance and behavior of the plot
fig9 = plot_datetime_data(
    df=loads_df,
    labels=Labels(y1_label="Load in MW"),
    line_styles=LineStyles(line_style1=StyleName.DASH),
    line_widths=LineWidths(line_width1=3),
    font_sizes=FontSizes(legend_fs=15),
)
# fig9.show()

# plot_datetime_comparison
# plot_datetime_comparison is built on the top of plot_datetime_data. It accepts a filtered_df that is illustrated in
# dark colors, while the dropped values are shown in light colors.
# Note that plot_datetime_comparison can accept all parameters that plot_datetime_data can accept e.g. labels, axes,
# color_mapper, etc.
mask_high_site_loads = loads_df["Total Site Load"] > 8  # MW
fig10 = plot_datetime_comparison(
    df=loads_df,
    filtered_df=loads_df[mask_high_site_loads],
)
fig10.show()
