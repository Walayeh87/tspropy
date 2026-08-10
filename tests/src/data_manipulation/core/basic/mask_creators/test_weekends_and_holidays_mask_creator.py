import pandas as pd
import pytest
from pandas import DatetimeIndex, Series

from src.data_manipulation.core.basic.mask_creators.weekends_and_holidays_mask_creator import (
    create_holiday_mask,
    create_weekend_mask,
)
from src.data_manipulation.custom_objects.location import Location


@pytest.mark.parametrize(
    "index, location, expected_holiday_mask",
    [
        (
            pd.to_datetime(["2024-01-01", "2024-01-02"]),
            Location(country_code="DE"),
            Series([True, False], index=pd.to_datetime(["2024-01-01", "2024-01-02"]), dtype=bool),
        ),
        (
            pd.date_range(start="2024-01-01", end="2024-01-02", freq="D", tz="UTC"),
            Location(country_code="DE", subdivision="Thüringen"),
            Series(
                [True, False],
                index=pd.date_range(start="2024-01-01", end="2024-01-02", freq="D", tz="UTC"),
                dtype=bool,
            ),
        ),
        (
            pd.to_datetime([]),
            Location(country_code="DE", subdivision="Thüringen"),
            Series([], index=pd.to_datetime([]), dtype=bool),
        ),
    ],
)
def test_create_holiday_mask(index: DatetimeIndex, location: Location, expected_holiday_mask: Series) -> None:
    holiday_mask = create_holiday_mask(index=index, location=location)

    assert isinstance(holiday_mask, Series)
    assert holiday_mask.index.equals(index)
    assert holiday_mask.dtype == bool
    assert holiday_mask.equals(expected_holiday_mask)


@pytest.mark.parametrize(
    "index,  expected_weekend_mask",
    [
        (
            pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02"]),
            Series(
                [False, False, True, True],
                index=pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-01", "2026-08-02"]),
                dtype=bool,
            ),
        ),
        (
            pd.to_datetime([]),
            Series(
                [],
                index=pd.to_datetime([]),
                dtype=bool,
            ),
        ),
    ],
)
def test_create_weekend_mask(index: DatetimeIndex, expected_weekend_mask: Series) -> None:
    weekend_mask = create_weekend_mask(index=index)

    assert isinstance(weekend_mask, Series)
    assert weekend_mask.index.equals(index)
    assert weekend_mask.dtype == bool
    assert weekend_mask.equals(expected_weekend_mask)
