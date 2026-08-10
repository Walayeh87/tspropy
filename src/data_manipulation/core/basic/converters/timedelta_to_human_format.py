import pandas as pd
from pandas import NaT, Timedelta

from src.data_manipulation.constants.time_constant import TimeConstant


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


if __name__ == "__main__":
    td1 = Timedelta(days=2, hours=5, minutes=30, seconds=5, milliseconds=7)
    td2 = Timedelta(hours=3, minutes=45)
    td3 = Timedelta(minutes=15)
    td4 = Timedelta(seconds=0)

    print(convert_timedelta_to_human_format(td1))  # Output: "2d 5h 30min 5s"
    print(convert_timedelta_to_human_format(td2))  # Output: "3h 45min"
    print(convert_timedelta_to_human_format(td3))  # Output: "15min"
    print(convert_timedelta_to_human_format(td4))  # Output: "0min"
