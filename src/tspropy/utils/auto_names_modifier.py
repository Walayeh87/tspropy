from enum import Enum


class AutoNamesModifier(str, Enum):
    """
    Base Enum class where:
    - auto() generates a lowercase string value from the member name
    - underscores are replaced with spaces
    - str() returns the value for seamless usage as strings
    """

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        return name.lower().replace("_", " ")

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)
