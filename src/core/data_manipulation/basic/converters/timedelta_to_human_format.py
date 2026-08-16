import pandas as pd
from pandas import NaT, Timedelta

from src.constants.time_constant import TimeConstant


def convert_timedelta_to_human_format(timedelta: Timedelta | NaT) -> str:
    if pd.isna(timedelta):
        raise ValueError("'timedelta' must not be NaT.")

    total_seconds = int(timedelta.total_seconds())

    days = total_seconds // TimeConstant.SEC_IN_DAY
    hours = (total_seconds % TimeConstant.SEC_IN_DAY) // TimeConstant.SEC_IN_H
    minutes = (total_seconds % TimeConstant.SEC_IN_H) // TimeConstant.SEC_IN_MIN
    seconds = total_seconds % TimeConstant.SEC_IN_MIN

    parts = []

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if seconds:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0min"
