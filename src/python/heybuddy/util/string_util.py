import re
from math import modf
from typing import Tuple, Union, Optional, Sequence

__all__ = [
    "reduce_units",
    "human_size",
    "human_duration",
    "safe_name",
]

def reduce_units(
    value: Union[int, float],
    units: Sequence[Union[str, Tuple[str, Union[int, float]]]],
    base: Union[int, float] = 1000,
) -> Tuple[float, str]:
    """
    Reduce a value to the smallest unit possible.

    >>> reduce_units(4e9, ["bytes/s", "kb/s", "mb/s", "gb/s"])
    (4.0, 'gb/s')

    :param value: The value to reduce.
    :param units: A sequence of units to reduce the value to.
                  Each unit can be a string or a tuple of (unit, base).
    :param base: The base to use for the units.
    :return: A tuple of the reduced value and the unit.
    """
    try:
        unit = units[0]
    except IndexError:
        raise ValueError("At least one unit must be provided.")

    for unit_or_tuple in units:
        if isinstance(unit_or_tuple, tuple):
            unit, unit_base = unit_or_tuple
        else:
            unit = unit_or_tuple
            unit_base = base
        if value < unit_base:
            break
        value /= unit_base
    return value, unit # type: ignore[return-value]

def human_size(
    num_bytes: Union[int, float],
    base_2: bool = False,
    precision: int = 2
) -> str:
    """
    Convert a number of bytes to a human-readable string.

    >>> human_size(1000)
    '1.00 KB'
    >>> human_size(1000**3)
    '1.00 GB'
    >>> human_size(1024, base_2=True)
    '1.00 KiB'
    >>> human_size(1024**3, base_2=True)
    '1.00 GiB'

    :param num_bytes: The number of bytes to convert.
    :param base_2: Whether to use base-2 units (e.g., KiB) or base-10 units (e.g., KB).
    :param precision: The number of decimal places to include in the output.
    :return: A human-readable string representing the number of bytes.
    """
    if base_2:
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        divisor = 1024.0
    else:
        units = ["B", "KB", "MB", "GB", "TB"]
        divisor = 1000.0

    reduced_bytes, unit = reduce_units(num_bytes, units, base=divisor)

    return f"{reduced_bytes:.{precision}f} {unit}"

def human_duration(
    duration_s: Union[int, float],
    precision: Optional[float] = None,
) -> str:
    """
    Convert a number of seconds to a human-readable string.
    Decimal precision is variable.

    Value < 1 second:
        Nanoseconds, microseconds, and milliseconds are reported as integers.
    1 second < value < 1 minute:
        Seconds are reported as floats with one decimal place.
    1 minute < value < 1 hour:
        Reported as minutes and seconds in the format "<x> m <y> s" with no decimal places.
    1 hour < value < 1 day:
        Reported as hours and minutes in the format "<x> h <y> m <z> s" with no decimal places.
    1 day < value:
        Reported as days and hours in the format "<x> d <y> h <z> m <zz> s" with no decimal places.

    >>> human_duration(0.00001601)
    '16 µs'
    >>> human_duration(1.5)
    '1.5 s'
    >>> human_duration(65)
    '1 m 5 s'
    >>> human_duration(3665)
    '1 h 1 m 5 s'
    >>> human_duration(90065)
    '1 d 1 h 1 m 5 s'

    :param duration_s: The duration in seconds to convert.
    :param precision: The number of decimal places to include in the output.
    :return: A human-readable string representing the duration.
    """
    # First set the duration to nanoseconds
    duration_s *= 1e9
    units = ["ns", "µs", "ms", "s", "m", "h", "d"]
    bases = [1e3, 1e3, 1e3, 60, 60, 24, 1000]
    reduced_seconds, unit = reduce_units(
        duration_s,
        list(zip(units, bases)),
        base=1000,
    )
    if unit in ["d", "h", "m"]:
        # Split the seconds into a whole part and a fractional part
        fractional, whole = modf(reduced_seconds)
        whole_formatted = f"{whole:.0f} {unit}"
        if fractional == 0:
            return whole_formatted
        # Return the fractional part to seconds
        if unit in ["d", "h", "m"]:
            fractional *= 60
        if unit in ["d", "h"]:
            fractional *= 60
        if unit == "d":
            fractional *= 24
        return " ".join([
            whole_formatted,
            human_duration(fractional, precision=0)
        ])
    else:
        if unit in ["ns", "µs", "ms"] and precision is None:
            precision = 1 if reduced_seconds < 10 else 0
        elif unit == "s" and precision is None:
            precision = 1
        return f"{reduced_seconds:.{precision}f} {unit}"

def safe_name(name: str) -> str:
    """
    Convert a string to a safe filename by removing invalid characters.
    """
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    name = re.sub(r"_+", "_", name)
    return name.strip("_")
