import time

import numpy as np
import pandas as pd

from src.data_manipulation.custom_objects.location import Location
from src.data_manipulation.custom_objects.plotter_objects import (
    Axes,
    BuiltinRange,
    Color,
    ColorMapper,
    CustomRange,
    Labels,
    LineWidths,
    MapperOfShadedAreasBounds,
    RangeConfig,
    RangePropertyName,
    Style,
    StyleName,
    Transparency,
    VLineConfig,
    VLinePropertyName,
    YLimits,
)
from src.plotters.plotly_plotters import plot_datetime_comparison, plot_datetime_data

np.random.seed(42)  # You can use any integer value
periods = 148
df = pd.DataFrame(
    {
        "col1": np.random.randint(0, 100, periods),
        "col2": np.random.randint(0, 90, periods),
        "col3": [np.nan] * periods,
        "col4": np.random.randint(0, 80, periods),
    },
    index=pd.date_range(
        start="1/1/2020", periods=periods, freq="60min"
    ),  # Index can be a naive range --> this line can be removed
)

color_mapper = [
    ColorMapper(curve_name="col1", color=Color.RED),
    # ColorMapper(curve_name="col2", color=Color.BLUE),
    ColorMapper(curve_name="col3", color=Color.GREEN),
    ColorMapper(curve_name="col4", color=Color.ORANGE),
]
mask = df["col1"] > 50
start_time = time.time()
v_line_config = [
    VLineConfig(
        position=df.index.floor("D").unique(),
        property={
            VLinePropertyName.COLOR: Color.BLACK,
            VLinePropertyName.WIDTH: 2,
            VLinePropertyName.DASH: Style(name=StyleName.DASHDOT),
        },
    ),
    VLineConfig(
        position=df.index.floor("3h").unique(),
        property={
            VLinePropertyName.COLOR: Color.RED,
            VLinePropertyName.WIDTH: 1,
            VLinePropertyName.DASH: Style(name=StyleName.SOLID),
        },
    ),
]
range_config = [
    RangeConfig(
        range_type=CustomRange(range=["2020-01-02 00:33", "2020-01-02 05:43"]),
        property={
            RangePropertyName.COLOR: Color.GREEN,
            RangePropertyName.TRANSPARENCY: Transparency(value=0.3),
        },
    ),
    RangeConfig(
        range_type=BuiltinRange.WEEKENDS,
        property={
            RangePropertyName.COLOR: Color.PURPLE,
            RangePropertyName.TRANSPARENCY: Transparency(value=0.1),
        },
    ),
    # RangeConfig(MaskRange(range=mask)),
    RangeConfig(BuiltinRange.HOLIDAYS, location=Location("DE")),
]

fig1 = plot_datetime_data(
    df=df,
    axes=Axes(y1=["col1"], y2=["col2", "col3"], y3=["col4"]),
    labels=Labels(y1_label="Values1", y2_label="Values2", y3_label="Values3"),
    y_limits=YLimits(y1_limit=None, y2_limit=[-5, 115], y3_limit=[-55, 190]),
    color_mapper=color_mapper,
    # range_config=range_config,
    v_line_config=v_line_config,
    line_widths=LineWidths(line_width1=2, line_width2=1),
)

fig1.show(renderer="browser")

end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")

# ==========================

filtered_df = df[mask]
fig2 = plot_datetime_comparison(
    df=df,
    filtered_df=filtered_df,
    axes=Axes(y1=["col1"], y2=["col2", "col3"], y3=["col4"]),
    labels=Labels(y1_label="Values1", y2_label="Values2", y3_label="Values3"),
)

fig2.show()


df["col1_upper_bound"] = df["col1"] + 10
df["col1_lower_bound"] = df["col1"] - 10
df["col2_upper_bound"] = df["col2"] * 1.1
df["col2_lower_bound"] = df["col2"] - 1.3

fig3 = plot_datetime_data(
    df=df,
    axes=Axes(y1=["col1"], y2=["col2", "col3"], y3=["col4"]),
    mapper_of_shaded_areas=[
        MapperOfShadedAreasBounds(
            curve_name="col1", upper_bound_name="col1_upper_bound", lower_bound_name="col1_lower_bound"
        ),
        MapperOfShadedAreasBounds(
            curve_name="col2", upper_bound_name="col2_upper_bound", lower_bound_name="col2_lower_bound"
        ),
    ],
    transparency_of_shaded_areas=Transparency(value=0.2),
)
fig3.show()
