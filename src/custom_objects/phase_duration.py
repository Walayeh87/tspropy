from dataclasses import dataclass

from pandas import Timedelta

from src.utils.internal_checkers import ensure_positive_timedelta


@dataclass
class PositiveTimedelta:
    value: str | Timedelta

    def __post_init__(self) -> None:
        ensure_positive_timedelta(timedelta=self.value)
