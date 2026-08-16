class DataframeDimensionError(Exception):
    """It is used when a df with wrong dimensions is detected."""


class EmptyDataError(Exception):
    """It is used when a df or series is empty."""


class NonMatchingIndexesError(Exception):
    """It is used when 2 objects have non-matching indexes."""


class NonMatchingColumnsError(Exception):
    """It is used when 2 objects have non-matching columns."""


class NonBooleanSeriesError(Exception):
    """It is used when a pandas Series is not a boolean one."""


class InvalidFreqError(Exception):
    """It is used when the frequency is invalid."""


class InvalidTimestampError(Exception):
    """It is used when the timestamp is invalid."""


class InvalidTimedeltaError(Exception):
    """It is used when the timedelta is invalid."""


class ExecutionError(Exception):
    """It is used when an error occurs during the execution of the main.py/cli.py."""
