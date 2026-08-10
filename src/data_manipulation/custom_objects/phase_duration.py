from dataclasses import dataclass

import pandas as pd

from src.data_manipulation.utils.internal_checkers import ensure_positive_timedelta


@dataclass
class PhaseDuration:
    value: str | pd.Timedelta

    def __post_init__(self) -> None:
        ensure_positive_timedelta(timedelta=self.value)
