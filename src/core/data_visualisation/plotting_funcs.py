import logging
from collections import Counter
from collections.abc import Hashable
from dataclasses import fields

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame, DatetimeIndex, Series
from pandas.core.dtypes.common import is_numeric_dtype
from plotly.graph_objs import Figure

from src.core.data_manipulation.basic.converters.series_and_frame import convert_frame2series
from src.core.data_manipulation.basic.df_processing.data_updater import update_data
from src.core.data_manipulation.basic.df_processing.row_droppers import drop_rows_with_duplicated_indices
from src.core.data_manipulation.basic.list_processing.list_processing import convert_to_flat_list
from src.core.data_manipulation.basic.mask_creators.weekends_and_holidays_mask_creator import create_holiday_mask
from src.core.data_manipulation.basic.mask_processing.mask_properties import (
    get_mask_on_starts,
    get_mask_on_stops,
    get_mask_phase_durations_as_ints,
    get_mask_phase_numbers,
)
from src.custom_objects.location import Location
from src.custom_objects.plotter_objects import (
    Axes,
    BuiltinRange,
    Color,
    ColorMapper,
    CustomRange,
    Data2Plot,
    FontSizes,
    HoverMode,
    Labels,
    LineStyles,
    LineWidths,
    MapperOfShadedAreasBounds,
    MaskRange,
    RangeConfig,
    RangePropertyName,
    Transparency,
    VLineConfig,
    YLimits,
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def plot_datetime_data(
    # Primary params
    df: DataFrame,
    axes: Axes | None = None,
    title: str = "",
    labels: Labels | None = None,
    y_limits: YLimits | None = None,
    color_mapper: ColorMapper | list[ColorMapper] | None = None,
    # Advanced features
    range_config: RangeConfig | list[RangeConfig] | None = None,
    v_line_config: VLineConfig | list[VLineConfig] | None = None,
    mapper_of_shaded_areas: MapperOfShadedAreasBounds | list[MapperOfShadedAreasBounds] | None = None,
    transparency_of_shaded_areas: Transparency | None = None,
    # Secondary features
    line_styles: LineStyles | None = None,
    line_widths: LineWidths | None = None,
    font_sizes: FontSizes | None = None,
    show_legend: bool = True,
    show_grid: bool = True,
    hover_mode: HoverMode = HoverMode.unified,
    y2_position: float | int = 0.88,
    title_margin: int = 230,
    # Params only used with plot_datetime_comparison
    filtered_df: DataFrame | None = None,
    data2plot: Data2Plot = Data2Plot.ALL,
    transparency_of_curves: Transparency | None = None,
    input_fig: Figure | None = None,
) -> Figure:
    """
    Create a Plotly time-series figure for datetime-indexed tabular df with flexible axis mapping,
    annotation of vertical/horizontal ranges, shaded bounds and styling options.

    This is the primary plotting entrypoint used to visualize time series df stored in a pandas
    DataFrame whose index is a DatetimeIndex. It supports mapping columns to up to three Y-axes,
    highlighting built-in ranges (weekends, holidays, AM/PM), custom or mask-based ranges, plotting
    vertical lines, and showing shaded areas between specified upper/lower bound columns.

    High-level behavior
    - Normalizes and validates provided parameters (axes, color mapping, ranges, etc.).
    - Optionally filters/updates df if `filtered_df` and `data2plot` are provided (used for comparison).
    - Assigns default colors, line styles, widths and font sizes if not supplied.
    - Adds traces to a Plotly `Figure` object (creates a new one if `input_fig` is None).
    - Adds rectangular shapes for highlighted ranges and vertical line shapes.
    - Updates layout (axes, legend, hover mode) and returns the final `Figure`.

    Args:
        df (DataFrame):
            The main time-series dataset to plot. Must be indexed by a DatetimeIndex. Each column
            represents a curve that can be plotted on one of the axes.
        axes (Axes | None, optional):
            Object that specifies which columns go to y1, y2 and y3 axes. If None, a default `Axes`
            is created and all columns are plotted on y1.
            See `src.data_manipulation.custom_objects.plotter_objects.Axes`.
        title (str, optional):
            Title text for the plot.
        labels (Labels | None, optional):
            Label texts for x, y1, y2 and y3. Defaults assigned when omitted. See `Labels`.
        y_limits (YLimits | None, optional):
            Explicit y-axis limits for y1, y2 and y3 — if omitted, limits are computed from the provided `df`.
        color_mapper (ColorMapper | list[ColorMapper] | None, optional):
            Color assignment(s) for curves. Can be a single `ColorMapper`, a list thereof, or None.
            If None or incomplete, defaults will be assigned for unlisted curves. ColorMappers must
            provide unique colors and unique curve names (duplicates raise ValueError).
        range_config (RangeConfig | list[RangeConfig] | None, optional):
            Which time ranges to highlight. Each `RangeConfig` may be a builtin (weekends, holidays,
            AM/PM), a `CustomRange` (explicit start/end), or a `MaskRange` (boolean mask). See
            `RangeConfig` and `BuiltinRange`. For holiday highlighting, `RangeConfig.location` must be set.
        v_line_config (VLineConfig | list[VLineConfig] | None, optional):
            Vertical line(s) to draw. Each VLineConfig contains one or more positions and line style props.
        mapper_of_shaded_areas (MapperOfShadedAreasBounds | list[MapperOfShadedAreasBounds] | None, optional):
            One or more mappers that define pairs of columns (upper/lower bound names) to be plotted as
            shaded areas around a base curve. Useful for confidence intervals, error bands, etc.
        transparency_of_shaded_areas (Transparency | None, optional):
            Global transparency for shaded areas. Defaults to a value if omitted.
        line_styles (LineStyles | None, optional):
            Object which holds dash style strings for each axis (line_style1/2/3). Defaults assigned when None.
        line_widths (LineWidths | None, optional):
            Widths for lines for each axis. Defaults assigned when None.
        font_sizes (FontSizes | None, optional):
            Font size settings for title, labels, legend and ticks. Defaults assigned when None.
        show_legend (bool, optional):
            Whether to display the legend.
        show_grid (bool, optional):
            Whether to show gridlines for x/y axes (applies to primary layout behavior).
        hover_mode (HoverMode, optional):
            Controls Plotly hover behavior. Default is `HoverMode.unified`. For comparison plots,
            `HoverMode.closest` is typically used.
        y2_position (float | int, optional):
            Relative horizontal position (domain coordinate) for the second y-axis when present.
        input_fig (Figure | None, optional):
            An existing Plotly `Figure` to add traces/shapes to. If None, a new `go.Figure()` is created.
            This enables building layered plots (see `plot_datetime_comparison` usage).
        title_margin (float | int, optional):
            Margin (in pixels) for the title.
        filtered_df (DataFrame | None, optional):
            An alternate DataFrame (same columns and columns order as `df`) representing filtered rows
            (for example, the rows removed by some mask). Used in combination with `data2plot` to plot
            dropped or remaining df. When supplied, `filtered_df.index` must be a subset of `df.index`.
        data2plot (Data2Plot, optional):
            Enum controlling which dataset to plot when `filtered_df` is provided:
            - Data2Plot.ALL: plot original `df` unchanged.
            - Data2Plot.DROPPED: plot the rows that were dropped (requires `filtered_df`).
            - Data2Plot.REMAINING: plot the remaining rows (requires `filtered_df`).
        transparency_of_curves (Transparency | None, optional):
            Alpha value applied to curve colors. Defaults to fully opaque when omitted.

    Returns:
        Figure:
            A Plotly `Figure` with added Scatter traces, shaded area traces (if requested) and layout
            shapes for highlighted ranges and vertical lines. The returned figure is suitable for
            display in notebooks, Plotly servers, or to be saved as an image.

    Raises:
        ValueError:
            - If one or more column names mapped in `axes` are not present in `df`.
            - If `filtered_df` is provided and:
                - its index is not a subset of `df`'s index
                - it has a different number of columns, or different column names/order
            - If duplicate colors or duplicate curve names are specified in `color_mapper`.
            - If a `RangeConfig` other than BuiltinRange.HOLIDAYS supplies a `location`.
            - If `mapper_of_shaded_areas` is provided but its type is not supported.
        TypeError:
            - If `color_mapper`, `range_config`, or `v_line_config` are passed in an unsupported type.
        NotImplementedError:
            - If an unsupported `data2plot` value is used.
            - If a `range_config` value refers to an unsupported BuiltinRange.

    Notes:
        - When `axes` is None, all DataFrame columns are plotted on the first axis.
        - When `mapper_of_shaded_areas` is supplied, shaded-area mappers extend the set of columns considered
          when computing axis limits and plotting (the bound column names must exist in `df`).
        - Ranges created from boolean masks only highlight continuous phases of length >= 2 index points.
        - The function relies on several internal helper functions (prefixed with `_`) to:
            - normalize inputs (_process_params / converters)
            - validate parameters (_validate_params)
            - compute axis limits (_calculate_y_limit)
            - construct shapes for ranges and vertical lines (_get_v_line_and_range_shapes)
            - add traces and shaded areas to the figure (_plot_all_axes, _plot_on_axis, _plot_shaded_area)
        - Colors may be provided as members of the `Color` enum; if not provided, defaults from `Color`
          enumeration are assigned. Each curve must have a unique color and name in the final mapping.
    """
    if axes is None:
        axes = Axes()
        axes.y1 = list(df.columns)

    axes, color_mappers, range_configs, v_line_configs = _process_params(
        df=df,
        axes=axes,
        color_mapper=color_mapper,
        range_config=range_config,
        v_line_config=v_line_config,
    )
    _validate_params(
        df=df, filtered_df=filtered_df, axes=axes, color_mappers=color_mappers, range_configs=range_configs
    )

    color_mappers = _fill_missing_inputs_in_color_mappers(
        all_cols=list(df),
        default_colors=list(Color),
        color_mappers=color_mappers,
        specified_curves=axes.y1 + axes.y2 + axes.y3,
    )

    labels = _assign_default_labels(labels=labels)
    y_limits = _assign_default_y_limits(y_limits=y_limits)
    line_styles = _assign_default_line_styles(line_styles=line_styles)
    line_widths = _assign_default_line_widths(widths=line_widths)
    font_sizes = _assign_default_font_size_values(font_sizes=font_sizes)
    transparency_of_curves, transparency_of_shaded_areas = _assign_default_transparency_values(
        transparency_of_curves=transparency_of_curves, transparency_of_shaded_areas=transparency_of_shaded_areas
    )

    original_data = df.copy()
    df = _update_data(df=df, original_data=original_data, filtered_df=filtered_df, data2plot=data2plot)

    mappers_of_shaded_areas = _convert_mapper_of_shaded_areas_to_list(mapper_of_shaded_areas=mapper_of_shaded_areas)
    df = _drop_unused_cols(df=df, y1=axes.y1, y2=axes.y2, y3=axes.y3, mappers_of_shaded_areas=mappers_of_shaded_areas)
    extended_y1, extended_y2, extended_y3 = _get_extended_axes(
        mappers_of_shaded_areas=mappers_of_shaded_areas, y1=axes.y1, y2=axes.y2, y3=axes.y3
    )

    y1_limit = _calculate_y_limit(extended_y=extended_y1, y_limit=y_limits.y1_limit, original_data=original_data)
    y2_limit = _calculate_y_limit(extended_y=extended_y2, y_limit=y_limits.y2_limit, original_data=original_data)
    y3_limit = _calculate_y_limit(extended_y=extended_y3, y_limit=y_limits.y3_limit, original_data=original_data)

    fig = _select_figure2use(input_fig=input_fig)

    _plot_all_axes(
        df=df,
        color_mappers=color_mappers,
        fig=fig,
        legendgroup1="legendgroup1",
        legendgroup2="legendgroup2",
        legendgroup3="legendgroup3",
        mappers_of_shaded_areas=mappers_of_shaded_areas,
        showlegend=show_legend,
        transparency_of_curves=transparency_of_curves,
        transparency_of_shaded_areas=transparency_of_shaded_areas,
        y1=axes.y1,
        y2=axes.y2,
        y3=axes.y3,
        yaxis1="y1",
        yaxis2="y2",
        yaxis3="y3",
        line_style1=line_styles.line_style1,
        line_style2=line_styles.line_style2,
        line_style3=line_styles.line_style3,
        line_width1=line_widths.line_width1,
        line_width2=line_widths.line_width2,
        line_width3=line_widths.line_width3,
    )

    if y1_limit is None:
        y1_limit = [float(df.min().min()), float(df.max().max())]

    v_line_and_range_shapes = _get_v_line_and_range_shapes(
        range_configs=range_configs, v_line_configs=v_line_configs, y1_lim=y1_limit, index=df.index
    )

    _update_layout(
        fig=fig,
        shapes=v_line_and_range_shapes,
        title=title,
        x_label=labels.x_label,
        y1_label=labels.y1_label,
        y1_limit=y1_limit,
        y2=axes.y2,
        y2_label=labels.y2_label,
        y2_limit=y2_limit,
        y2_position=y2_position,
        y3=axes.y3,
        y3_label=labels.y3_label,
        y3_limit=y3_limit,
        hovermode=hover_mode,
        title_fs=font_sizes.title_fs,
        label_fs=font_sizes.label_fs,
        legend_fs=font_sizes.legend_fs,
        xy_ticks_fs=font_sizes.xy_ticks_fs,
        grid=show_grid,
        title_margin=title_margin,
    )

    return fig


def plot_datetime_comparison(
    df: DataFrame,
    filtered_df: DataFrame,
    transparency_of_filtered_phases: Transparency | None = None,
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ANN003
) -> Figure:
    if transparency_of_filtered_phases is None:
        transparency_of_filtered_phases = Transparency(value=0.4)

    fig1 = plot_datetime_data(
        df=df,
        filtered_df=filtered_df,
        data2plot=Data2Plot.DROPPED,
        transparency_of_curves=transparency_of_filtered_phases,
        show_legend=False,
        hover_mode=HoverMode.closest,
        *args,  # noqa: B026
        **kwargs,
    )

    df = df.copy()

    fig2 = plot_datetime_data(
        df=df,
        filtered_df=filtered_df,
        data2plot=Data2Plot.REMAINING,
        transparency_of_curves=Transparency(value=1),
        input_fig=fig1,
        show_legend=True,
        hover_mode=HoverMode.closest,
        *args,  # noqa: B026
        **kwargs,
    )

    return fig2


def _get_numerical_columns(df: DataFrame) -> list:
    numerical_cols = []
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            numerical_cols.append(column)

    return numerical_cols


def _assign_default_line_widths(widths: LineWidths | None) -> LineWidths:
    if widths is None:
        widths = LineWidths()

    return widths


def _assign_default_line_styles(line_styles: LineStyles | None) -> LineStyles:
    if line_styles is None:
        line_styles = LineStyles()

    return line_styles


def _assign_default_y_limits(y_limits: YLimits | None) -> YLimits:
    if y_limits is None:
        y_limits = YLimits()

    return y_limits


def _assign_default_labels(labels: Labels | None) -> Labels:
    if labels is None:
        labels = Labels()

    return labels


def _select_figure2use(input_fig: Figure | None) -> Figure:
    if input_fig is None:
        input_fig = go.Figure()

    return input_fig


def _process_params(
    df: DataFrame,
    axes: Axes,
    color_mapper: ColorMapper | list[ColorMapper] | None,
    range_config: RangeConfig | list[RangeConfig] | None,
    v_line_config: VLineConfig | list[VLineConfig] | None,
) -> tuple[Axes, list[ColorMapper], list[RangeConfig], list[VLineConfig]]:
    axes = _drop_nan_columns(df=df, axes=axes)
    color_mappers = _convert_color_mapper_to_list(color_mapper=color_mapper)
    range_configs = _convert_range_config_to_list(range_config=range_config)
    v_line_configs = _convert_v_line_config_to_list(v_line_config=v_line_config)

    return axes, color_mappers, range_configs, v_line_configs


def _validate_params(
    df: DataFrame,
    filtered_df: DataFrame | None,
    axes: Axes,
    color_mappers: list[ColorMapper],
    range_configs: list[RangeConfig],
) -> None:
    numerical_cols = _get_numerical_columns(df=df)
    if len(numerical_cols) < len(df.columns):
        non_numerical_cols = [col for col in df.columns if col not in numerical_cols]
        df = df[numerical_cols]
        if all(item in axes.y1 for item in non_numerical_cols):
            axes.y1 = [col for col in axes.y1 if col not in non_numerical_cols]
        logger.warning(f"Non-numerical columns {non_numerical_cols} were removed from the DataFrame; ")

    if len(axes.y1 + axes.y2 + axes.y3) > len(list(Color)):
        raise ValueError(
            f"plot_datetime_data can only plot {len(list(Color))} curves."
            f" Provided curves: {len(axes.y1 + axes.y2 + axes.y3)}"
        )

    for y in [axes.y1 + axes.y2 + axes.y3]:
        for col in y:
            if col not in df.columns:
                raise ValueError(f"'{col}' is not in the provided df!")

    if filtered_df is not None:
        if not filtered_df.index.isin(df.index).all():
            raise ValueError("filtered_df must be a subset of the original df.")

        if len(df.columns) != len(filtered_df.columns):
            raise ValueError("The number of columns in filtered_df must match those in the original df.")

        if all(df.columns) != all(filtered_df.columns):
            raise ValueError("The columns in filtered_df must match those in the original df.")

    all_colors = [color_mapper.color for color_mapper in color_mappers]
    duplicate_colors = [item for item, count in Counter(all_colors).items() if count > 1]
    if len(duplicate_colors) > 0:
        raise ValueError(
            f"Duplicate colors found in color_mapper! Each curve should have a unique color."
            f" Duplicate colors: {duplicate_colors}"
        )

    all_cols = [color_mapper.curve_name for color_mapper in color_mappers]
    duplicate_cols = [item for item, count in Counter(all_cols).items() if count > 1]
    if len(duplicate_cols) > 0:
        raise ValueError(
            f"Duplicate curve names found in color_mapper! Each curve should have a unique name."
            f" Duplicate curve names: {duplicate_cols}"
        )

    for range_config in range_configs:
        if range_config.range_type != BuiltinRange.HOLIDAYS and range_config.location is not None:
            raise ValueError(
                f"location must be specified only for plotting public holidays."
                f" Please remove the location parameter for '{range_config.range_type}'."
            )


def _calculate_y_limit(
    extended_y: list, y_limit: list[int | float] | None, original_data: DataFrame
) -> list[int | float] | None:
    if isinstance(y_limit, list):
        return y_limit

    factor_to_get_curves_away_from_borders = 0.05

    used_data = original_data[extended_y]

    if used_data.empty:
        return None

    y_min = np.nanmin(used_data.min())
    y_max = np.nanmax(used_data.max())
    y_range = y_max - y_min

    return [
        y_min - y_range * factor_to_get_curves_away_from_borders,
        y_max + y_range * factor_to_get_curves_away_from_borders,
    ]


def _get_line_style(axis_number: int, line_style1: str, line_style2: str, line_style3: str) -> str:
    if axis_number == 1:
        return line_style1
    elif axis_number == 2:
        return line_style2
    elif axis_number == 3:
        return line_style3
    else:
        raise ValueError(f"The axis_number must be 1, 2 or 3, got {axis_number}!")


def _get_line_width(
    axis_number: int, line_width1: float | int, line_width2: float | int, line_width3: float | int
) -> float | int:
    if axis_number == 1:
        return line_width1
    elif axis_number == 2:
        return line_width2
    elif axis_number == 3:
        return line_width3
    else:
        raise ValueError(f"The axis_number must be 1, 2 or 3, got {axis_number}!")


def _drop_nan_columns(df: DataFrame, axes: Axes) -> Axes:
    for col in axes.y1:
        if df[col].isna().all():
            logger.warning(f"The column '{str(col)}' contains only NaN values, and thus, it will be ignored!")
            axes.y1.remove(col)

    for col in axes.y2:
        if df[col].isna().all():
            logger.warning(f"The column '{str(col)}' contains only NaN values, and thus, it will be ignored!")
            axes.y2.remove(col)

    for col in axes.y3:
        if df[col].isna().all():
            logger.warning(f"The column '{str(col)}' contains only NaN values, and thus, it will be ignored!")
            axes.y3.remove(col)

    return axes


def _get_v_line_and_range_shapes(
    range_configs: list[RangeConfig],
    v_line_configs: list[VLineConfig],
    y1_lim: list[int | float],
    index: DatetimeIndex,
) -> list:
    v_line_shapes = _create_shapes_for_v_lines(v_line_configs=v_line_configs)
    range_shapes = _highlight_range(range_configs=range_configs, y1_lim=y1_lim, index=index)

    return convert_to_flat_list(list_of_lists=range_shapes + v_line_shapes)


def _plot_all_axes(
    df: DataFrame,
    color_mappers: list[ColorMapper],
    fig: Figure,
    legendgroup1: str,
    legendgroup2: str,
    legendgroup3: str,
    mappers_of_shaded_areas: list[MapperOfShadedAreasBounds],
    showlegend: bool,
    transparency_of_curves: Transparency,
    transparency_of_shaded_areas: Transparency,
    y1: list,
    y2: list,
    y3: list,
    yaxis1: str,
    yaxis2: str,
    yaxis3: str,
    line_style1: str,
    line_style2: str,
    line_style3: str,
    line_width1: int | float,
    line_width2: int | float,
    line_width3: int | float,
) -> None:
    if len(y1) == 0 and len(y2) == 0 and len(y3) == 0:
        logger.info("Since no axis was specified, all df columns are going to be plotted on the first axis.")
        _plot_on_axis(
            df=df,
            axis_number=1,
            number_of_axes=1,
            y=list(df),
            yaxis=yaxis1,
            legendgroup=legendgroup1,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            transparency_of_curves=transparency_of_curves,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )

    elif len(y1) != 0 and len(y2) == 0 and len(y3) == 0:
        _plot_on_axis(
            df=df,
            axis_number=1,
            number_of_axes=1,
            y=y1,
            yaxis=yaxis1,
            legendgroup=legendgroup1,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )

    elif len(y1) != 0 and len(y2) != 0 and len(y3) == 0:
        _plot_on_axis(
            df=df,
            axis_number=1,
            number_of_axes=2,
            y=y1,
            yaxis=yaxis1,
            legendgroup=legendgroup1,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )
        _plot_on_axis(
            df=df,
            axis_number=2,
            number_of_axes=2,
            y=y2,
            yaxis=yaxis2,
            legendgroup=legendgroup2,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )

    elif len(y1) != 0 and len(y2) != 0 and len(y3) != 0:
        _plot_on_axis(
            df=df,
            axis_number=1,
            number_of_axes=3,
            y=y1,
            yaxis=yaxis1,
            legendgroup=legendgroup1,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )
        _plot_on_axis(
            df=df,
            axis_number=2,
            number_of_axes=3,
            y=y2,
            yaxis=yaxis2,
            legendgroup=legendgroup2,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )
        _plot_on_axis(
            df=df,
            axis_number=3,
            number_of_axes=3,
            y=y3,
            yaxis=yaxis3,
            legendgroup=legendgroup3,
            color_mappers=color_mappers,
            fig=fig,
            transparency_of_curves=transparency_of_curves,
            transparency_of_shaded_areas=transparency_of_shaded_areas,
            mappers_of_shaded_areas=mappers_of_shaded_areas,
            showlegend=showlegend,
            line_style1=line_style1,
            line_style2=line_style2,
            line_style3=line_style3,
            line_width1=line_width1,
            line_width2=line_width2,
            line_width3=line_width3,
        )

    else:
        raise ValueError("y1, y2 then y3 should be specified. Check your parameters!")


def _update_data(
    df: DataFrame, original_data: DataFrame, filtered_df: DataFrame | None, data2plot: Data2Plot
) -> DataFrame:
    if data2plot == Data2Plot.DROPPED and filtered_df is not None:
        dropped_data = _get_dropped_data(df=df, filtered_df=filtered_df)
        mask_nan_starts = get_mask_on_starts(mask=dropped_data[dropped_data.columns[0]].isna())
        mask_nan_ends = get_mask_on_stops(mask=dropped_data[dropped_data.columns[0]].isna())
        return update_data(data=dropped_data, slices=original_data[mask_nan_starts | mask_nan_ends])
    elif data2plot == Data2Plot.REMAINING and filtered_df is not None:
        return _get_remaining_data(df=df, filtered_df=filtered_df)
    elif data2plot == Data2Plot.ALL:
        return df
    else:
        raise NotImplementedError(
            f"data2plot value '{data2plot}' is not supported. Supported values: '{[e.value for e in Data2Plot]}'"
        )


def _get_remaining_data(df: DataFrame, filtered_df: DataFrame) -> DataFrame:
    return filtered_df.reindex(df.index)


def _fill_missing_inputs_in_color_mappers(
    all_cols: list, default_colors: list, color_mappers: list[ColorMapper], specified_curves: list
) -> list[ColorMapper]:
    if len(color_mappers) == 0:
        _fill_unspecified_with_default_colors(
            all_cols=all_cols, default_colors=default_colors, color_mappers=color_mappers
        )

    color_mappers = _remove_items_belong_to_shaded_areas(color_mappers=color_mappers, specified_curves=specified_curves)

    assigned_colors = [color_mapper.color for color_mapper in color_mappers]
    assigned_curves = [color_mapper.curve_name for color_mapper in color_mappers]
    if (len(assigned_colors) != len(specified_curves)) or (len(assigned_colors) == 0):
        color_mappers = _assign_colors(
            all_cols=all_cols,
            default_colors=default_colors,
            color_mappers=color_mappers,
            assigned_colors=assigned_colors,
            assigned_curves=assigned_curves,
        )

    return color_mappers


def _get_dropped_data(df: DataFrame, filtered_df: DataFrame) -> Series | DataFrame:
    dropped_data = pd.concat([df, filtered_df], axis="index")
    dropped_data = drop_rows_with_duplicated_indices(data=dropped_data, index2keep=False)
    dropped_data = dropped_data.sort_index()

    return dropped_data.reindex(df.index)


def _drop_unused_cols(
    df: DataFrame,
    y1: list,
    y2: list,
    y3: list,
    mappers_of_shaded_areas: list[MapperOfShadedAreasBounds],
) -> DataFrame:
    data_cols = list(df)

    all_mapper_values = []
    for mapper in mappers_of_shaded_areas:
        mapper_values = [getattr(mapper, field.name) for field in fields(mapper)]
        all_mapper_values.append(mapper_values)

    if len(y1) == 0 and len(y2) == 0 and len(y3) == 0:
        cols2use = data_cols + all_mapper_values
        cols2use = list(set(cols2use))

        return df[cols2use]

    specified_cols = y1 + y2 + y3
    cols2use = specified_cols + all_mapper_values

    cols2use = convert_to_flat_list(list_of_lists=cols2use)
    cols2use = list(set(cols2use))

    return df[cols2use]


def _highlight_range(range_configs: list[RangeConfig], y1_lim: list[int | float], index: DatetimeIndex) -> list:
    shapes = []
    for range_config in range_configs:
        if isinstance(range_config.range_type, BuiltinRange):
            if range_config.range_type == BuiltinRange.WEEKENDS:
                weekend_shapes = _create_weekend_range_shapes(
                    index=index,
                    y1_lim=y1_lim,
                    range_properties=range_config.property,
                )
                shapes.append(weekend_shapes)

            elif range_config.range_type == BuiltinRange.HOLIDAYS:
                holiday_shapes = []
                if range_config.location is not None:
                    holiday_shapes = _create_holiday_range_shapes(
                        index=index,
                        y1_lim=y1_lim,
                        range_properties=range_config.property,
                        location=range_config.location,
                    )
                shapes.append(holiday_shapes)

            elif range_config.range_type == BuiltinRange.AM:
                am_shapes = _create_am_range_shapes(index=index, y1_lim=y1_lim, range_properties=range_config.property)
                shapes.append(am_shapes)

            elif range_config.range_type == BuiltinRange.PM:
                pm_shapes = _create_pm_range_shapes(index=index, y1_lim=y1_lim, range_properties=range_config.property)
                shapes.append(pm_shapes)

            else:
                raise ValueError(
                    f"Unsupported range value: {range_config.range_type}. Please choose one of {list(BuiltinRange)}"
                )

        elif isinstance(range_config.range_type, CustomRange):
            custom_range_shapes = _create_custom_range_shapes(
                range2highlight=range_config.range_type.range, range_properties=range_config.property, y1_lim=y1_lim
            )
            shapes.append(custom_range_shapes)

        elif isinstance(range_config.range_type, MaskRange):
            mask_as_range = _convert_mask_to_ranges(mask=range_config.range_type.range)
            mask_range_shapes = _create_mask_range_shapes(
                range2highlight=mask_as_range, range_properties=range_config.property, y1_lim=y1_lim
            )
            shapes.append(mask_range_shapes)

        else:
            raise NotImplementedError(
                f"The range type {range_config.range_type} is not implemented."
                f" Supported types: {CustomRange} and {BuiltinRange}"
            )

    return shapes


def _create_weekend_range_shapes(
    index: DatetimeIndex, y1_lim: list[int | float], range_properties: dict[RangePropertyName, Color | Transparency]
) -> list:
    days = index.floor("d").unique()
    saturdays = list(days[days.weekday == 5])

    return _create_day_range_shapes(
        index=index,
        y1_lim=y1_lim,
        range_properties=range_properties,
        days_to_start_highlighting_from=saturdays,
        length_in_days=2,
    )


def _create_holiday_range_shapes(
    index: DatetimeIndex,
    y1_lim: list[int | float],
    range_properties: dict[RangePropertyName, Color | Transparency],
    location: Location,
) -> list:
    is_holiday = create_holiday_mask(index=index, location=location)
    holidays = list(index[is_holiday].floor("d").unique())

    return _create_day_range_shapes(
        index=index,
        y1_lim=y1_lim,
        range_properties=range_properties,
        days_to_start_highlighting_from=holidays,
        length_in_days=1,
    )


def _create_day_range_shapes(
    index: DatetimeIndex,
    y1_lim: list[int | float],
    range_properties: dict[RangePropertyName, Color | Transparency],
    days_to_start_highlighting_from: list,
    length_in_days: int,
) -> list:
    return [
        dict(
            type="rect",
            x0=day,
            x1=day + pd.Timedelta(days=length_in_days)
            if day + pd.Timedelta(days=length_in_days) <= index.max()
            else index.max(),
            y0=y1_lim[0],
            y1=y1_lim[1],
            fillcolor=range_properties.get(RangePropertyName.COLOR),
            opacity=range_properties.get(RangePropertyName.TRANSPARENCY),
            layer="below",
            line_width=0,
        )
        for day in days_to_start_highlighting_from
    ]


def _create_am_range_shapes(
    index: DatetimeIndex, y1_lim: list[int | float], range_properties: dict[RangePropertyName, Color | Transparency]
) -> list:
    is_midnight = index.hour == 0

    return _create_hour_range_shapes(
        index=index,
        y1_lim=y1_lim,
        range_properties=range_properties,
        hours_to_start_highlighting_from=list(index[is_midnight].floor("h").unique()),
    )


def _create_pm_range_shapes(
    index: DatetimeIndex, y1_lim: list[int | float], range_properties: dict[RangePropertyName, Color | Transparency]
) -> list:
    is_noon = index.hour == 12

    return _create_hour_range_shapes(
        index=index,
        y1_lim=y1_lim,
        range_properties=range_properties,
        hours_to_start_highlighting_from=list(index[is_noon].floor("h").unique()),
    )


def _create_hour_range_shapes(
    index: DatetimeIndex,
    y1_lim: list[int | float],
    range_properties: dict[RangePropertyName, Color | Transparency],
    hours_to_start_highlighting_from: list,
) -> list:
    return [
        dict(
            type="rect",
            x0=point,
            x1=point + pd.Timedelta(hours=12) if point + pd.Timedelta(hours=12) <= index.max() else index.max(),
            y0=y1_lim[0],
            y1=y1_lim[1],
            fillcolor=range_properties.get(RangePropertyName.COLOR),
            opacity=range_properties.get(RangePropertyName.TRANSPARENCY),
            layer="below",
            line_width=0,
        )
        for point in hours_to_start_highlighting_from
    ]


def _update_layout(
    fig: Figure,
    shapes: list,
    hovermode: str,
    title: str | None,
    x_label: str | None,
    y1_label: str | None,
    y1_limit: list[int | float],
    y2: list[str | Hashable] | None,
    y2_label: str | None,
    y2_limit: list[int | float] | None,
    y2_position: float | int | None,
    y3: list[str | Hashable] | None,
    y3_label: str | None,
    y3_limit: list[int | float] | None,
    title_fs: int | float,
    label_fs: int | float,
    legend_fs: int | float,
    xy_ticks_fs: int | float,
    title_margin: int,
    grid: bool,
) -> None:
    title_properties = dict(
        text=title,
        font=dict(size=title_fs),
        y=0.99,
        x=0.5,
    )
    x_axis_properties = dict(
        title=dict(text=x_label, font=dict(size=label_fs)),
        showgrid=grid,
        domain=[0, y2_position],  # if y2 or y3 else [0, 1],
        tickfont=dict(size=xy_ticks_fs),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="gray",
        ticks="outside",
        ticklen=8,
        tickwidth=2,
    )
    y1_axis_properties = dict(
        title=dict(text=y1_label, font=dict(size=label_fs)),
        showgrid=False if y2 or y3 else grid,
        range=y1_limit if isinstance(y1_limit, list) else None,
        tickfont=dict(size=xy_ticks_fs),
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=False if y2 or y3 else True,
        gridcolor="gray",
        ticks="outside",
        ticklen=8,
        tickwidth=2,
    )
    y2_axis_properties = dict(
        title=dict(text=y2_label, font=dict(size=label_fs)),
        anchor="free",
        overlaying="y",
        side="right",
        showgrid=False,
        position=y2_position,
        range=y2_limit if isinstance(y2_limit, list) else None,
        tickfont=dict(size=xy_ticks_fs),
        showline=True,
        linewidth=2,
        linecolor="black",
        ticks="outside",
        ticklen=8,
        tickwidth=2,
    )
    y3_axis_properties = dict(
        title=dict(text=y3_label, font=dict(size=label_fs)),
        anchor="free",
        overlaying="y",
        side="right",
        showgrid=False,
        position=1,
        range=y3_limit if isinstance(y3_limit, list) else None,
        tickfont=dict(size=xy_ticks_fs),
        showline=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        ticklen=8,
        tickwidth=2,
    )
    legend_properties = dict(
        orientation="h",
        yanchor="bottom",
        x=0.5,
        y=1.02,
        xanchor="center",
        traceorder="grouped",
        font_size=legend_fs,
        bgcolor=_apply_alpha("black", 0.1),
    )

    fig.update_traces(hovertemplate="<b>%{fullData.name}</b>: %{y:.2f}<extra></extra>")

    fig.update_layout(
        title=title_properties,
        xaxis=x_axis_properties,
        yaxis=y1_axis_properties,
        yaxis2=y2_axis_properties,
        yaxis3=y3_axis_properties,
        legend=legend_properties,
        yaxis_showgrid=False if y2 or y3 else grid,
        yaxis2_showgrid=False,
        yaxis3_showgrid=False,
        yaxis_zeroline=False,
        yaxis2_zeroline=False,
        yaxis3_zeroline=False,
        hovermode=hovermode,
        hoverlabel=dict(font_size=16, namelength=1),
        plot_bgcolor="white",  # Background inside the plot area
        margin=dict(t=title_margin),  # Add top margin for title
        shapes=shapes,
    )


def _plot_on_axis(
    df: DataFrame,
    axis_number: int,
    number_of_axes: int,
    y: list[str | Hashable],
    yaxis: str,
    legendgroup: str,
    color_mappers: list[ColorMapper],
    fig: Figure,
    transparency_of_shaded_areas: Transparency,
    transparency_of_curves: Transparency,
    showlegend: bool,
    mappers_of_shaded_areas: list[MapperOfShadedAreasBounds],
    line_style1: str,
    line_style2: str,
    line_style3: str,
    line_width1: int | float,
    line_width2: int | float,
    line_width3: int | float,
) -> None:
    _validate_axis_number(axis_number=axis_number)
    _validate_number_of_axes(number_of_axes=number_of_axes)

    line_style = _get_line_style(
        axis_number=axis_number, line_style1=line_style1, line_style2=line_style2, line_style3=line_style3
    )
    line_width = _get_line_width(
        axis_number=axis_number, line_width1=line_width1, line_width2=line_width2, line_width3=line_width3
    )

    for col in y:
        corresponding_color = next(mapper.color.name for mapper in color_mappers if mapper.curve_name == col)
        trace = go.Scatter(
            x=df.index,
            y=df[col],
            name=col,
            yaxis=yaxis,
            legendgroup=legendgroup,
            line=dict(
                width=line_width,
                dash=line_style,
                color=_apply_alpha(color=corresponding_color, alpha=transparency_of_curves.value),
            ),
            showlegend=showlegend,
        )

        if len(mappers_of_shaded_areas) == 0:
            fig.add_trace(trace)
            continue

        cols_in_mapper = [mapper.curve_name for mapper in mappers_of_shaded_areas]
        if col in cols_in_mapper:
            corresponding_shaded_area_mapper = next(
                mapper for mapper in mappers_of_shaded_areas if mapper.curve_name == col
            )
            corresponding_color = next(mapper for mapper in color_mappers if mapper.curve_name == col)
            _plot_shaded_area(
                df=df,
                col=col,
                yaxis=yaxis,
                color_mapper=corresponding_color,
                mapper_of_shaded_areas=corresponding_shaded_area_mapper,
                fig=fig,
                transparency_of_shaded_areas=transparency_of_shaded_areas,
            )

        fig.add_trace(trace)


def _plot_shaded_area(
    df: DataFrame,
    col: str,
    yaxis: str,
    color_mapper: ColorMapper,
    mapper_of_shaded_areas: MapperOfShadedAreasBounds,
    fig: Figure,
    transparency_of_shaded_areas: Transparency,
) -> None:
    # Add shaded area (upper bound)
    color = _apply_alpha(color_mapper.color.name, transparency_of_shaded_areas.value)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[mapper_of_shaded_areas.upper_bound_name],
            mode="lines",
            name="Upper Bound of " + str(col),
            fillcolor=color,
            line=dict(color=color),
            showlegend=False,
            yaxis=yaxis,
        )
    )

    # Add shaded area (lower bound) and fill the area between lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[mapper_of_shaded_areas.lower_bound_name],
            mode="lines",
            name="Lower Bound of " + str(col),
            fillcolor=color,
            line=dict(color=color),
            showlegend=False,
            yaxis=yaxis,
            fill="tonexty",  # Fills the space between this and the previous trace
        )
    )


def _fill_unspecified_with_default_colors(
    all_cols: list, default_colors: list, color_mappers: list[ColorMapper]
) -> None:
    for color_index, curve_name in enumerate(all_cols):
        color_mappers.append(ColorMapper(curve_name=curve_name, color=default_colors[color_index]))


def _assign_colors(
    all_cols: list, default_colors: list, color_mappers: list, assigned_colors: list, assigned_curves: list
) -> list[ColorMapper]:
    unassigned_curves = [curve for curve in all_cols if curve not in assigned_curves]
    unassigned_colors = [color for color in default_colors if color not in assigned_colors]

    if len(unassigned_curves) > len(unassigned_colors):
        unassigned_curves = unassigned_curves[: len(unassigned_colors)]
    if len(unassigned_colors) > len(unassigned_curves):
        unassigned_colors = unassigned_colors[: len(unassigned_curves)]

    for curve, color in zip(unassigned_curves, unassigned_colors, strict=True):
        color_mappers.append(ColorMapper(curve_name=curve, color=color))

    return color_mappers


def _apply_alpha(color: str, alpha: float | int = 1.0) -> str:
    rgba = _convert_color_with_alpha_into_rgba(color, alpha)

    return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"


def _convert_color_with_alpha_into_rgba(
    color: str,
    alpha: float | int,
) -> tuple[float | int, float | int, float | int, float | int]:
    return mcolors.to_rgba(color, alpha)


def _create_shapes_for_v_lines(v_line_configs: list[VLineConfig]) -> list:
    shapes = []
    for v_line_config in v_line_configs:
        for x in v_line_config.position:
            shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line=v_line_config.property))

    return shapes


def _remove_items_belong_to_shaded_areas(color_mappers: list[ColorMapper], specified_curves: list) -> list[ColorMapper]:
    return [color_mapper for color_mapper in color_mappers if color_mapper.curve_name in specified_curves]


def _convert_color_mapper_to_list(color_mapper: ColorMapper | list[ColorMapper] | None) -> list[ColorMapper]:
    if isinstance(color_mapper, list):
        return color_mapper
    elif isinstance(color_mapper, ColorMapper):
        return [color_mapper]
    elif color_mapper is None:
        return []
    else:
        raise TypeError(
            f"color_mapper should be either None, a ColorMapper object or a list of ColorMapper objects!"
            f" Got {type(color_mapper)} instead."
        )


def _convert_range_config_to_list(range_config: RangeConfig | list[RangeConfig] | None) -> list[RangeConfig]:
    if isinstance(range_config, list):
        return range_config
    elif isinstance(range_config, RangeConfig):
        return [range_config]
    elif range_config is None:
        return []
    else:
        raise TypeError(
            f"range_config should be either None, a RangeConfig object or a list of RangeConfig objects!"
            f" Got {type(range_config)} instead."
        )


def _convert_v_line_config_to_list(v_line_config: VLineConfig | list[VLineConfig] | None) -> list[VLineConfig]:
    if isinstance(v_line_config, list):
        return v_line_config
    elif isinstance(v_line_config, VLineConfig):
        return [v_line_config]
    elif v_line_config is None:
        return []
    else:
        raise TypeError(
            f"v_line_config should be either None, a VLineConfig object or a list of VLineConfig objects!"
            f" Got {type(v_line_config)} instead."
        )


def _convert_mask_to_ranges(mask: Series[bool] | DataFrame) -> list:
    mask = convert_frame2series(data=mask)

    mask = mask.fillna(False)
    phase_nrs = get_mask_phase_numbers(mask=mask)
    phase_durs = get_mask_phase_durations_as_ints(mask=mask)

    positive_phase_nrs = phase_nrs[phase_nrs > 0]
    positive_phase_nrs_longer_than_one = positive_phase_nrs[phase_durs > 1]

    ranges = []
    for _, g in mask.groupby(positive_phase_nrs_longer_than_one):
        ranges.append((g.index[0], g.index[-1]))

    return ranges


def _create_mask_range_shapes(
    range2highlight: list, range_properties: dict[RangePropertyName, Color | Transparency], y1_lim: list[int | float]
) -> list:
    shapes = []

    for start, end in range2highlight:
        shapes.append(
            dict(
                type="rect",
                x0=start,
                x1=end,
                y0=y1_lim[0],
                y1=y1_lim[1],
                fillcolor=range_properties.get(RangePropertyName.COLOR),
                opacity=range_properties.get(RangePropertyName.TRANSPARENCY),
                layer="below",
                line_width=0,
            )
        )

    return shapes


def _create_custom_range_shapes(
    range2highlight: list, range_properties: dict[RangePropertyName, Color | Transparency], y1_lim: list[int | float]
) -> list:
    return [
        dict(
            type="rect",
            x0=range2highlight[0],
            x1=range2highlight[1],
            y0=y1_lim[0],
            y1=y1_lim[1],
            fillcolor=range_properties.get(RangePropertyName.COLOR),
            opacity=range_properties.get(RangePropertyName.TRANSPARENCY),
            layer="below",
            line_width=0,
        )
    ]


def _validate_axis_number(axis_number: int) -> None:
    if axis_number not in [1, 2, 3]:
        raise ValueError(f"The axis_number must be 1, 2 or 3! {axis_number} was passed.")


def _validate_number_of_axes(number_of_axes: int) -> None:
    if number_of_axes not in [1, 2, 3]:
        raise ValueError(f"The number_of_axes must be 1, 2 or 3! {number_of_axes} was passed.")


def _get_extended_axes(
    mappers_of_shaded_areas: list[MapperOfShadedAreasBounds], y1: list, y2: list, y3: list
) -> tuple[list[str], list[str], list[str]]:
    extended_y1 = y1
    extended_y2 = y2
    extended_y3 = y3
    for mapper in mappers_of_shaded_areas:
        cols_in_mapper = [getattr(mapper, field.name) for field in fields(mapper)]
        if any(item in cols_in_mapper for item in y1):
            extended_y1 = list(set(y1 + cols_in_mapper))
        if any(item in cols_in_mapper for item in y2):
            extended_y2 = list(set(y2 + cols_in_mapper))
        if any(item in cols_in_mapper for item in y3):
            extended_y3 = list(set(y3 + cols_in_mapper))

    return extended_y1, extended_y2, extended_y3


def _convert_mapper_of_shaded_areas_to_list(
    mapper_of_shaded_areas: MapperOfShadedAreasBounds | list[MapperOfShadedAreasBounds] | None,
) -> list[MapperOfShadedAreasBounds]:
    if mapper_of_shaded_areas is None:
        return []
    elif isinstance(mapper_of_shaded_areas, MapperOfShadedAreasBounds):
        return [mapper_of_shaded_areas]
    elif isinstance(mapper_of_shaded_areas, list):
        return mapper_of_shaded_areas
    else:
        raise ValueError(
            f"mapper_of_shaded_areas should be either a MapperOfShadedAreasBounds object or"
            f" a list of MapperOfShadedAreasBounds objects! Got {type(mapper_of_shaded_areas)} instead."
        )


def _assign_default_transparency_values(
    transparency_of_curves: Transparency | None, transparency_of_shaded_areas: Transparency | None
) -> tuple[Transparency, Transparency]:
    if transparency_of_shaded_areas is None:
        transparency_of_shaded_areas = Transparency(value=0.2)

    if transparency_of_curves is None:
        transparency_of_curves = Transparency(value=1)

    return transparency_of_curves, transparency_of_shaded_areas


def _assign_default_font_size_values(font_sizes: FontSizes | None) -> FontSizes:
    if font_sizes is None:
        font_sizes = FontSizes()

    return font_sizes
