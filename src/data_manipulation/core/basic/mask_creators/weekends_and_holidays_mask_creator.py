import holidays
import pandas as pd
from pandas import DatetimeIndex, Series

from src.data_manipulation.custom_objects.location import Location


def create_holiday_mask(index: DatetimeIndex, location: Location) -> Series:
    """
    Returns a boolean mask indicating whether each timestamp falls on a public holiday for the given
    country and subdivision.

    Parameters
    ----------
    index : DatetimeIndex
        Time index to evaluate.
    location : Location
        Location object containing country code and optional subdivision.

    Returns
    -------
    Series[bool]
        True where timestamp is public holiday.
    """
    original_index = index.copy()

    years = original_index.year.unique()
    country_holidays = holidays.country_holidays(
        country=location.country_code, subdiv=location.subdivision, years=years
    )
    country_holidays = pd.to_datetime(list(country_holidays))
    index = original_index.tz_localize(None)
    is_holiday = index.normalize().isin(country_holidays)

    return Series(is_holiday, index=original_index)


def create_weekend_mask(index: DatetimeIndex) -> Series:
    """
    Returns a boolean mask indicating whether each timestamp falls on a weekend.

    Parameters
    ----------
    index : DatetimeIndex
        Time index to evaluate.

    Returns
    -------
    Series[bool]
        True where timestamp is weekend.
    """
    is_weekend = index.weekday >= 5

    return Series(is_weekend, index=index)
