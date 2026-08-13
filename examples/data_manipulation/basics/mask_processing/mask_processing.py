import pandas as pd

from src.data_manipulation.core.basic.mask_processing.mask_properties import (
    get_mask_phase_durations,
    get_mask_phase_numbers,
)
from src.data_manipulation.custom_objects.plotter_objects import (
    Axes,
    Color,
    ColorMapper,
    Labels,
    MaskRange,
    RangeConfig,
    RangePropertyName,
    YLimits,
)
from src.plotters.plotly_plotters import plot_datetime_data

df = pd.read_csv(
    r"../../../data/load_and_price.csv",
    index_col="datetime",
    parse_dates=True,
)

energy_color = Color.PURPLE
site_load_color = Color.RED

fig1 = plot_datetime_data(
    df=df[["Total Site Load"]],
    labels=Labels(y1_label="Load in MW"),
    y_limits=YLimits(y1_limit=[5, 10]),
    color_mapper=ColorMapper("Total Site Load", site_load_color),
    title="Total Site Load Overview",
)
# fig1.show(renderer="browser")

# get_mask_phase_numbers() use case
# Introduction about the function and its importance
mask_high_consumption = df["Total Site Load"] > 8  # MW

high_consumption_phase_nr = get_mask_phase_numbers(mask=mask_high_consumption)
data_freq_in_min = 5
min_in_h = 60

calculate_energy = lambda power: sum(power * data_freq_in_min / min_in_h)
df["Phase Energy"] = df.groupby(high_consumption_phase_nr)["Total Site Load"].transform(func=calculate_energy)


fig2 = plot_datetime_data(
    df=df,
    axes=Axes(y1="Total Site Load", y2="Phase Energy"),
    labels=Labels(y1_label="Load in MW", y2_label="Energy in MWh"),
    color_mapper=[
        ColorMapper("Total Site Load", site_load_color),
        ColorMapper("Phase Energy", energy_color),
    ],
    range_config=RangeConfig(
        range_type=MaskRange(range=mask_high_consumption),
        property={RangePropertyName.COLOR: energy_color},
    ),
    title="Phase-Based Energy Consumption",
)
# fig2.show(renderer="browser")

# get_mask_phase_durations() use case
# get_mask_phase_durations() is helpful for filtering data by its phase durations
load_phase_durations = get_mask_phase_durations(mask=mask_high_consumption)
mask_long_phases = load_phase_durations > pd.Timedelta("2h")

fig3 = plot_datetime_data(
    df=df,
    axes=Axes(y1="Total Site Load", y2="Phase Energy"),
    labels=Labels(y1_label="Load in MW", y2_label="Energy in MWh"),
    color_mapper=[
        ColorMapper("Total Site Load", site_load_color),
        ColorMapper("Phase Energy", energy_color),
    ],
    range_config=RangeConfig(
        range_type=MaskRange(range=mask_long_phases),
        property={RangePropertyName.COLOR: energy_color},
    ),
    title="Phase-Based Energy Consumption",
)
fig3.show(renderer="browser")
