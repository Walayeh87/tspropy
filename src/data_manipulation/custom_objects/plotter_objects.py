from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto

import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series

from src.data_manipulation.custom_objects.location import Location
from src.data_manipulation.utils.auto_names_modifier import AutoNamesModifier
from src.data_manipulation.utils.internal_checkers import ensure_boolean_series, ensure_data_has_one_column


class Color(Enum):
    BLACK = auto()
    BLUE = auto()
    RED = auto()
    GREEN = auto()
    CYAN = auto()
    ORANGE = auto()
    MAGENTA = auto()
    YELLOW = auto()
    BROWN = auto()
    PURPLE = auto()
    CORAL = auto()
    FUCHSIA = auto()
    LIGHTGRAY = auto()


class Data2Plot(Enum):
    ALL = auto()
    REMAINING = auto()
    DROPPED = auto()


class BuiltinRange(Enum):
    WEEKENDS = auto()
    HOLIDAYS = auto()
    AM = auto()
    PM = auto()


class StyleName(AutoNamesModifier):
    SOLID = auto()
    DASH = auto()
    DASHDOT = auto()


class RangePropertyName(AutoNamesModifier):
    COLOR = auto()
    TRANSPARENCY = auto()


class VLinePropertyName(AutoNamesModifier):
    COLOR = auto()
    WIDTH = auto()
    DASH = auto()


class HoverMode(StrEnum):
    closest = "closest"
    unified = "x unified"


@dataclass
class Axes:
    y1: list[str | Hashable] | str | Hashable | None = None
    y2: list[str | Hashable] | str | Hashable | None = None
    y3: list[str | Hashable] | str | Hashable | None = None

    def __post_init__(self) -> None:
        self.y1: list[str | Hashable] = self.__convert_to_list(value=self.y1)
        self.y2: list[str | Hashable] = self.__convert_to_list(value=self.y2)
        self.y3: list[str | Hashable] = self.__convert_to_list(value=self.y3)

        if len(self.y1 + self.y2 + self.y3) != len(set(self.y1 + self.y2 + self.y3)):
            duplicates = [x for x in self.y1 + self.y2 + self.y3 if (self.y1 + self.y2 + self.y3).count(x) > 1]
            raise ValueError(f"Duplicated curves were provided: {set(duplicates)}!")

        # y1 should be filled, then y2 and lastly y3
        if len(self.y1) == 0 and (len(self.y2) > 0 or len(self.y3) > 0):
            raise ValueError("y1 should be filled before y2 and y3!")

        if len(self.y2) == 0 and len(self.y3) > 0:
            raise ValueError("y2 should be filled before y3!")

    @staticmethod
    def __convert_to_list(value: list[str | Hashable] | str | Hashable | None) -> list[str | Hashable]:
        if value is None:
            return []
        if isinstance(value, str) or isinstance(value, Hashable):
            return [value]
        return value


@dataclass
class Transparency:
    value: int | float

    def __post_init__(self) -> None:
        if self.value == 0:
            raise ValueError("Transparency value cannot be 0! (invisible line!)")

        if self.value > 1:
            raise ValueError("Transparency value cannot be greater than 1!")

        if self.value < 0:
            raise ValueError("Transparency value cannot be negative!")


@dataclass
class Labels:
    x_label: str = "Time"

    y1_label: str = ""
    y2_label: str = ""
    y3_label: str = ""

    def __post_init__(self) -> None:
        if len({self.y1_label, self.y2_label, self.y3_label}) < 3 and (self.y1_label or self.y2_label or self.y3_label):
            raise ValueError("y1_label, y2_label and y3_label must be different from each other!")


@dataclass
class YLimit:
    range: list[int | float]

    def __post_init__(self) -> None:
        if len(self.range) != 2:
            raise ValueError("y_lim must be a list with two elements!")

        if self.range[0] >= self.range[1]:
            raise ValueError("y_lim values are reversed or identical!")


@dataclass
class YLimits:
    y1_limit: YLimit | None = None
    y2_limit: YLimit | None = None
    y3_limit: YLimit | None = None


@dataclass
class LineWidth:
    value: int | float = 1.8

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError(f"Width value must be a positive number! Got {self.value} instead!")

    def __lt__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value >= other.value

    def __eq__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value == other.value

    def __ne__(self, other: int | float) -> bool:
        if not isinstance(other, LineWidth):
            return NotImplemented
        return self.value != other.value


@dataclass
class LineWidths:
    line_width1: LineWidth | None = None
    line_width2: LineWidth | None = None
    line_width3: LineWidth | None = None

    def __post_init__(self) -> None:
        if self.line_width1 is None:
            self.line_width1 = LineWidth(value=1.8)

        if self.line_width2 is None:
            self.line_width2 = LineWidth(value=1.8)

        if self.line_width3 is None:
            self.line_width3 = LineWidth(value=1.8)


@dataclass
class Style:
    name: StyleName


@dataclass
class LineStyles:
    line_style1: StyleName = StyleName.SOLID
    line_style2: StyleName = StyleName.DASH
    line_style3: StyleName = StyleName.DASHDOT


@dataclass
class ColorMapper:
    curve_name: str | Hashable
    color: Color


@dataclass
class MapperOfShadedAreasBounds:
    curve_name: str
    upper_bound_name: str
    lower_bound_name: str

    def __post_init__(self) -> None:
        if len({self.curve_name, self.upper_bound_name, self.lower_bound_name}) < 3:
            raise ValueError("curve_name, upper_bound_name and lower_bound_name must be different from each other!")


@dataclass
class MaskRange:
    range: Series[bool] | DataFrame

    def __post_init__(self) -> None:
        ensure_data_has_one_column(data=self.range)

        if isinstance(self.range, Series):
            ensure_boolean_series(param=self.range)
        elif isinstance(self.range, DataFrame):
            ensure_boolean_series(param=self.range[self.range.columns[0]])
        else:
            raise TypeError(f"range must be a pandas Series or DataFrame! Got {type(self.range)} instead!")


@dataclass
class CustomRange:
    range: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.range, list):
            raise TypeError(f"range must be a list of two date strings! Got {type(self.range)} instead!")

        if len(self.range) != 2:
            raise ValueError("range must have 2 elements!")

        for item in self.range:
            try:
                pd.to_datetime(item)
            except ValueError as e:
                raise ValueError(f"Invalid range element '{item}'!") from e

        if pd.to_datetime(self.range[0]) >= pd.to_datetime(self.range[1]):
            raise ValueError("The range consists of identical or reversed elements!")


class DefaultValue(Enum):
    # Range
    DEFAULT_RANGE_COLOR = Color.BLACK
    DEFAULT_TRANSPARENCY = Transparency(value=0.2)

    # V-Line
    DEFAULT_VLINE_COLOR = Color.RED
    DEFAULT_STYLE = StyleName.SOLID
    DEFAULT_WIDTH = LineWidth(value=1.8)


@dataclass
class FontSizes:
    title_fs: int | float = 27
    label_fs: int | float = 27
    xy_ticks_fs: int | float = 24
    legend_fs: int | float = 24

    def __post_init__(self) -> None:
        if self.title_fs <= 0:
            raise ValueError(f"title_fs must be a positive number! Got {self.title_fs} instead!")

        if self.label_fs <= 0:
            raise ValueError(f"label_fs must be a positive number! Got {self.label_fs} instead!")

        if self.xy_ticks_fs <= 0:
            raise ValueError(f"xy_ticks_fs must be a positive number! Got {self.xy_ticks_fs} instead!")

        if self.legend_fs <= 0:
            raise ValueError(f"legend_fs must be a positive number! Got {self.legend_fs} instead!")


@dataclass
class VLineConfig:
    position: DatetimeIndex
    property: dict[VLinePropertyName, Color | Style | LineWidth]

    def __post_init__(self) -> None:
        if isinstance(self.property, dict):
            if len(self.property) > 3:
                raise ValueError("Range property can have at most 3 keys!")

            for key, value in self.property.items():
                if not isinstance(key, VLinePropertyName):
                    raise TypeError(f"property keys must be of type VLinePropertyName! Got {type(key)} instead!")

                if (key == VLinePropertyName.COLOR) and not isinstance(value, Color):
                    raise TypeError(f"'{key}' property value must be of type Color! Got {type(value)} instead!")

                if (key == VLinePropertyName.DASH) and not isinstance(value, Style):
                    raise TypeError(f"'{key}' property value must be of type Dash! Got {type(value)} instead!")

                if (key == VLinePropertyName.WIDTH) and not isinstance(value, LineWidth):
                    raise TypeError(f"'{key}' property value must be of type Width! Got {type(value)} instead!")

            self.__fill_missing_property_with_default_values()

        else:
            self.__fill_empty_property_with_default_values()

        self.__process_property_values()

    def __process_property_values(self) -> None:
        for key, value in self.property.items():
            if key == VLinePropertyName.COLOR:
                self.property.update({VLinePropertyName.COLOR: value.name})
            if key == VLinePropertyName.DASH:
                self.property.update({VLinePropertyName.DASH: value.name.lower()})
            if key == VLinePropertyName.WIDTH:
                self.property.update({VLinePropertyName.WIDTH: value.value})

    def __fill_missing_property_with_default_values(self) -> None:
        color = self.property.get(VLinePropertyName.COLOR)
        if color is None:
            self.property.update({VLinePropertyName.COLOR: DefaultValue.DEFAULT_VLINE_COLOR.value})

        dash = self.property.get(VLinePropertyName.DASH)
        if dash is None:
            self.property.update({VLinePropertyName.DASH: DefaultValue.DEFAULT_STYLE.value})

        width = self.property.get(VLinePropertyName.WIDTH)
        if width is None:
            self.property.update({VLinePropertyName.WIDTH: DefaultValue.DEFAULT_WIDTH.value})

    def __fill_empty_property_with_default_values(self) -> None:
        self.property = {}

        self.property.update({VLinePropertyName.COLOR: DefaultValue.DEFAULT_VLINE_COLOR.value})
        self.property.update({VLinePropertyName.DASH: DefaultValue.DEFAULT_STYLE.value})
        self.property.update({VLinePropertyName.WIDTH: DefaultValue.DEFAULT_WIDTH.value})


@dataclass
class RangeConfig:
    range_type: BuiltinRange | CustomRange | MaskRange
    property: dict[RangePropertyName, Color | Transparency] | None = None
    location: Location | None = None  # Only specified if range_type is BuiltinRange.HOLIDAYS!!

    def __post_init__(self) -> None:
        if (
            self.location is None
            and isinstance(self.range_type, BuiltinRange)
            and self.range_type == BuiltinRange.HOLIDAYS
        ):
            raise ValueError("location must be specified for plotting public holidays.")

        if isinstance(self.property, dict):
            if len(self.property) > 2:
                raise ValueError("Range property can have at most 2 keys!")

            for key, value in self.property.items():
                if not isinstance(key, RangePropertyName):
                    raise TypeError(f"Range property keys must be of type RangePropertyName! Got {type(key)} instead!")

                if (key == RangePropertyName.COLOR) and not isinstance(value, Color):
                    raise TypeError(f"'{key}' property value must be of type Color! Got {type(value)} instead!")

                if (key == RangePropertyName.TRANSPARENCY) and not isinstance(value, Transparency):
                    raise TypeError(f"'{key}' property value must be of type Transparency! Got {type(value)} instead!")

            self.__fill_missing_property_with_default_values()

        else:
            self.__fill_empty_property_with_default_values()

        self.__process_property_values()

    def __process_property_values(self) -> None:
        for key, value in self.property.items():
            if key == RangePropertyName.COLOR:
                self.property.update({RangePropertyName.COLOR: value.name})
            if key == RangePropertyName.TRANSPARENCY:
                self.property.update({RangePropertyName.TRANSPARENCY: value.value})

    def __fill_missing_property_with_default_values(self) -> None:
        color = self.property.get(RangePropertyName.COLOR)
        if color is None:
            self.property.update({RangePropertyName.COLOR: DefaultValue.DEFAULT_RANGE_COLOR.value})

        transparency = self.property.get(RangePropertyName.TRANSPARENCY)
        if transparency is None:
            self.property.update({RangePropertyName.TRANSPARENCY: DefaultValue.DEFAULT_TRANSPARENCY.value})

    def __fill_empty_property_with_default_values(self) -> None:
        self.property: dict = {}
        self.property.update({RangePropertyName.COLOR: DefaultValue.DEFAULT_RANGE_COLOR.value})
        self.property.update({RangePropertyName.TRANSPARENCY: DefaultValue.DEFAULT_TRANSPARENCY.value})
