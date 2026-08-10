import pytest

from src.data_manipulation.custom_objects.location import Location


def test_with_invalid_country_code() -> None:
    with pytest.raises(ValueError):
        Location(country_code="foo")


def test_with_invalid_subdivision() -> None:
    with pytest.raises(ValueError):
        Location(country_code="DE", subdivision="Damascus")

    with pytest.raises(ValueError):
        Location(country_code="DE", subdivision="HeSSen")  # the subdivision name is case-sensitive


def test_with_valid_entries() -> None:
    location1 = Location(country_code="DE", subdivision="Hessen")
    assert location1.country_code == "DE"
    assert location1.subdivision == "Hessen"

    location2 = Location(country_code="De", subdivision="Hessen")  # country code is case-insensitive
    assert location2.country_code == "DE"
    assert location2.subdivision == "Hessen"
