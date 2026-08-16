from dataclasses import dataclass

import pycountry


@dataclass
class Location:
    country_code: str  # case-insensitive, e.g. "DE" or "de"
    subdivision: str | None = None  # case-sensitive!

    def __post_init__(self) -> None:
        self.country_code = self.country_code.upper()

        country_codes = [country.alpha_2 for country in pycountry.countries]
        if self.country_code not in country_codes:
            raise ValueError(f"'{self.country_code}' is an invalid country code.")

        if self.subdivision is not None:
            subdivision_names = _get_subdivision_names(country_code=self.country_code)
            if self.subdivision not in subdivision_names:
                raise ValueError(
                    f"'{self.subdivision}' is an invalid subdivision name. Please check the spelling since it is "
                    f"case-sensitive. "
                    f" Valid subdivision names for country code '{self.country_code}' are: {subdivision_names}."
                )


def _get_subdivision_names(country_code: str) -> list:
    subdivision_keywords = {"Region", "Land", "Province", "State"}

    subdivision_names = []
    for subdivision in pycountry.subdivisions:
        is_matching_code = subdivision.country_code == country_code
        keyword_mentioned_in_subdivision_type = any(
            subdivision_keyword.lower() in subdivision.type.lower() for subdivision_keyword in subdivision_keywords
        )
        if is_matching_code and keyword_mentioned_in_subdivision_type:
            subdivision_names.append(subdivision.name)

    return sorted(subdivision_names)
