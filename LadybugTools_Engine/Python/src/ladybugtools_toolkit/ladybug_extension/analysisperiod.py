"""Methods for manipulating Ladybug analysis periods."""

# pylint: disable=E0401
import contextlib
import calendar
from datetime import datetime
from enum import Enum

# pylint: enable=E0401

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

from .dt import lb_datetime_from_datetime
from ._mapping import _MONTH_MAP, _TIMESTEP_FREQUENCY_MAP
from ..helpers import _NullWriter
from ..bhom.logger import CONSOLE_LOGGER

from python_toolkit.bhom.analytics import bhom_analytics


class DefaultAnalysisPeriod(Enum):
    """The set fo default analysis periods."""

    @staticmethod
    def _create_period(*args, **kwargs) -> AnalysisPeriod:
        """Create an AnalysisPeriod object without printing output."""
        with contextlib.redirect_stdout(_NullWriter()):
            return AnalysisPeriod(*args, **kwargs)

    # all hours
    ALL_HOURS = _create_period()

    # day periods
    MORNING = _create_period(st_hour=5, end_hour=12)
    AFTERNOON = _create_period(st_hour=13, end_hour=17)
    EVENING = _create_period(st_hour=18, end_hour=21)
    NIGHT = _create_period(st_hour=22, end_hour=4)

    # season periods
    DEC_FEB = _create_period(st_month=12, end_month=2)
    MAR_MAY = _create_period(st_month=3, end_month=5)
    JUN_AUG = _create_period(st_month=6, end_month=8)
    SEP_NOV = _create_period(st_month=9, end_month=11)

    # day + season periods
    DEC_FEB_MORNING = _create_period(st_month=12, end_month=2, st_hour=5, end_hour=12)
    DEC_FEB_AFTERNOON = _create_period(
        st_month=12, end_month=2, st_hour=13, end_hour=17
    )
    DEC_FEB_EVENING = _create_period(st_month=12, end_month=2, st_hour=18, end_hour=21)
    DEC_FEB_NIGHT = _create_period(st_month=12, end_month=2, st_hour=22, end_hour=4)
    MAR_MAY_MORNING = _create_period(st_month=3, end_month=5, st_hour=5, end_hour=12)
    MAR_MAY_AFTERNOON = _create_period(st_month=3, end_month=5, st_hour=13, end_hour=17)
    MAR_MAY_EVENING = _create_period(st_month=3, end_month=5, st_hour=18, end_hour=21)
    MAR_MAY_NIGHT = _create_period(st_month=3, end_month=5, st_hour=22, end_hour=4)
    JUN_AUG_MORNING = _create_period(st_month=6, end_month=8, st_hour=5, end_hour=12)
    JUN_AUG_AFTERNOON = _create_period(st_month=6, end_month=8, st_hour=13, end_hour=17)
    JUN_AUG_EVENING = _create_period(st_month=6, end_month=8, st_hour=18, end_hour=21)
    JUN_AUG_NIGHT = _create_period(st_month=6, end_month=8, st_hour=22, end_hour=4)
    SEP_NOV_MORNING = _create_period(st_month=9, end_month=11, st_hour=5, end_hour=12)
    SEP_NOV_AFTERNOON = _create_period(
        st_month=9, end_month=11, st_hour=13, end_hour=17
    )
    SEP_NOV_EVENING = _create_period(st_month=9, end_month=11, st_hour=18, end_hour=21)
    SEP_NOV_NIGHT = _create_period(st_month=9, end_month=11, st_hour=22, end_hour=4)

    def description(self, latitude: float) -> str:
        """Get the description of the analysis period.

        Args:
            latitude (float, optional):
                The latitude of the location.
                This is used to determine the season of the year.

        """
        if self.name not in [
            self.ALL_HOURS.name,
            self.MORNING.name,
            self.AFTERNOON.name,
            self.EVENING.name,
            self.NIGHT.name,
        ]:
            if latitude < 23.5 and latitude > -23.5:
                CONSOLE_LOGGER.warning(
                    "The latitude is in the tropics (between the Tropic of Cancer "
                    "and Tropic of Capricorn). Seasons are not as easily marked in "
                    "these regions, so they probably won't make much sense."
                )
        d = {
            self.ALL_HOURS.name: "Annual, all hours",
            self.MORNING.name: "Annual, morning",
            self.AFTERNOON.name: "Annual, afternoon",
            self.EVENING.name: "Annual, evening",
            self.NIGHT.name: "Annual, night",
            self.DEC_FEB.name: "Winter" if latitude > 0 else "Summer" + " (Dec-Feb)",
            self.MAR_MAY.name: "Spring" if latitude > 0 else "Autumn" + " (Mar-May)",
            self.JUN_AUG.name: "Summer" if latitude > 0 else "Winter" + " (Jun-Aug)",
            self.SEP_NOV.name: "Autumn" if latitude > 0 else "Spring" + " (Sep-Nov)",
            self.DEC_FEB_MORNING.name: ("Winter" if latitude > 0 else "Summer")
            + " morning (Dec-Feb, 05:00-12:59)",
            self.DEC_FEB_AFTERNOON.name: ("Winter" if latitude > 0 else "Summer")
            + " afternoon (Dec-Feb, 13:00-17:59)",
            self.DEC_FEB_EVENING.name: ("Winter" if latitude > 0 else "Summer")
            + " evening (Dec-Feb, 18:00-21:59)",
            self.DEC_FEB_NIGHT.name: ("Winter" if latitude > 0 else "Summer")
            + " night (Dec-Feb, 22:00-04:59)",
            self.MAR_MAY_MORNING.name: ("Spring" if latitude > 0 else "Autumn")
            + " morning (Mar-May, 05:00-12:59)",
            self.MAR_MAY_AFTERNOON.name: ("Spring" if latitude > 0 else "Autumn")
            + " afternoon (Mar-May, 13:00-17:59)",
            self.MAR_MAY_EVENING.name: ("Spring" if latitude > 0 else "Autumn")
            + " evening (Mar-May, 18:00-21:59)",
            self.MAR_MAY_NIGHT.name: ("Spring" if latitude > 0 else "Autumn")
            + " night (Mar-May, 22:00-04:59)",
            self.JUN_AUG_MORNING.name: ("Summer" if latitude > 0 else "Winter")
            + " morning (Jun-Aug, 05:00-12:59)",
            self.JUN_AUG_AFTERNOON.name: ("Summer" if latitude > 0 else "Winter")
            + " afternoon (Jun-Aug, 13:00-17:59)",
            self.JUN_AUG_EVENING.name: ("Summer" if latitude > 0 else "Winter")
            + " evening (Jun-Aug, 18:00-21:59)",
            self.JUN_AUG_NIGHT.name: ("Summer" if latitude > 0 else "Winter")
            + " night (Jun-Aug, 22:00-04:59)",
            self.SEP_NOV_MORNING.name: ("Autumn" if latitude > 0 else "Spring")
            + " morning (Sep-Nov, 05:00-12:59)",
            self.SEP_NOV_AFTERNOON.name: ("Autumn" if latitude > 0 else "Spring")
            + " afternoon (Sep-Nov, 13:00-17:59)",
            self.SEP_NOV_EVENING.name: ("Autumn" if latitude > 0 else "Spring")
            + " evening (Sep-Nov, 18:00-21:59)",
            self.SEP_NOV_NIGHT.name: ("Autumn" if latitude > 0 else "Spring")
            + " night (Sep-Nov, 22:00-04:59)",
        }

        return d[self.name]


def analysis_period_to_datetimes(
    analysis_period: AnalysisPeriod,
) -> pd.DatetimeIndex:
    """Convert an AnalysisPeriod object into a Pandas DatetimeIndex.

    Args:
        analysis_period (AnalysisPeriod):
            An AnalysisPeriod object.

    Returns:
        pd.DatetimeIndex:
            A Pandas DatetimeIndex object.
    """

    datetimes = pd.to_datetime(analysis_period.datetimes)

    return datetimes


@bhom_analytics()
def analysis_period_to_boolean(
    analysis_periods: list[AnalysisPeriod] | AnalysisPeriod,
) -> list[bool]:
    """Convert an AnalysisPeriod object into a list of booleans where values
    within the Period are also within a default whole analysis period of the
    same interval.

    Args:
        analysis_periods (list[AnalysisPeriod]):
            A list of AnalysisPeriod objects.

    Returns:
        list[bool]:
            A list of booleans
    """

    if isinstance(analysis_periods, AnalysisPeriod):
        analysis_periods = [analysis_periods]

    # check timestep of each analysis period is the same
    if len(set(ap.timestep for ap in analysis_periods)) > 1:
        raise ValueError("All analysis periods must have the same timestep.")

    # remove duplicates from list
    analysis_periods = list(set(analysis_periods))

    # create a generic set of datetimes for the same timestep
    generic_datetimes = analysis_period_to_datetimes(
        AnalysisPeriod(timestep=analysis_periods[0].timestep)
    )

    # for each analysis period in analysis_periods, create a list of booleans
    # where values within the Period are also within a default whole analysis
    # period of the same interval
    bools = []
    for ap in analysis_periods:
        bools.append(generic_datetimes.isin(analysis_period_to_datetimes(ap)))

    return np.any(bools, axis=0)


@bhom_analytics()
def analysis_period_from_datetimes(datetimes: list[datetime]) -> AnalysisPeriod:
    """Convert a list of datetimes (in order from earliest to latest) into an
    AnalysisPeriod object.

    Args:
        datetimes (list[datetime]):
            qA list of datetimes.

    Returns:
        AnalysisPeriod:
            An AnalysisPeriod object.
    """

    inferred_timestep = (60 * 60) / (datetimes[1] - datetimes[0]).seconds

    analysis_period = AnalysisPeriod.from_start_end_datetime(
        lb_datetime_from_datetime(min(datetimes)),
        lb_datetime_from_datetime(max(datetimes)),
        inferred_timestep,
    )

    if len(analysis_period.datetimes) != len(datetimes):
        raise ValueError(
            f"The number of datetimes ({len(datetimes)}) does not match the number of datetimes in "
            "the AnalysisPeriod ({len(analysis_period.datetimes)}), which probably means your "
            "datetime-list has an irregular time-step and cannot be coerced into an AnalysisPeriod."
        )
    return analysis_period


def analysis_period_to_string(analysis_period: AnalysisPeriod, save_path:bool = False) -> str:
    """Convert a Ladybug Analysis Period into a custom string representation.

    The resulting string may be converted back into an AnalysisPeriod object.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
            Defaults to False.

    Returns:
        str:
            A description of the analysis period.

    """
    if not isinstance(analysis_period, AnalysisPeriod):
        raise TypeError("analysis_period must be a Ladybug AnalysisPeriod object.")

    if save_path:
        base_str = (
            f"{analysis_period.st_month:02}{analysis_period.st_day:02}"
            f"_{analysis_period.end_month:02}{analysis_period.end_day:02}"
            f"_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"
            f"_{analysis_period.timestep:02}_{'L' if analysis_period.is_leap_year else 'C'}"
        )
        return base_str

    base_str = (
        f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to "
        f"{calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} "
        f"every {_TIMESTEP_FREQUENCY_MAP[analysis_period.timestep]} between "
        f"{analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:59"
        f" {'(L)' if analysis_period.is_leap_year else '(C)'}"
    )

    return base_str

def string_to_analysis_period(string: str) -> AnalysisPeriod:
    """Convert a custom string representation of an analysis period into a Ladybug AnalysisPeriod.

    Examples:
        >>> string = "0101_1231_0_23_1_L"
        >>> analysis_period = _string_to_analysis_period(string)
        >>> print(analysis_period)
        AnalysisPeriod(1, 1, 0, 12, 31, 23, 1, True)

        >>> string = "Mar 2 to Dec 31 between 04:00 and 22:59 every 30min (C)"
        >>> analysis_period = _string_to_analysis_period(string)
        >>> print(analysis_period)
        AnalysisPeriod(3, 2, 4, 12, 31, 22, 2, False)

    Args:
        string (str):
            The string representation of the analysis period.

    Returns:
        AnalysisPeriod: An object representing the analysis period.

    """
    try:
        pattern = (
            r"^(\d{2})(\d{2})_(\d{2})(\d{2})_(\d{1,2})_(\d{1,2})_(\d{1,2})_([LC])$"
        )
        match = re.match(pattern, string)
        (
            st_month,
            st_day,
            end_month,
            end_day,
            st_hour,
            end_hour,
            timestep,
            is_leap_year,
        ) = match.groups()  # type: ignore
    except AttributeError:
        try:
            # the string is in human readable format, try alternative method
            pattern = (
                r"^([A-Za-z]{3}) (\d{1,2}) to ([A-Za-z]{3}) (\d{1,2}) "
                r"every (\d+min|h|min) "
                r"between (\d{1,2}):(\d{1,2}) and (\d{1,2}):(\d{1,2}) "
                r"\(([LC])\)$"
            )
            match = re.match(pattern, string)
            (
                st_month,
                st_day,
                end_month,
                end_day,
                timestep,
                st_hour,
                _,
                end_hour,
                _,
                is_leap_year,
            ) = match.groups()  # type: ignore

            st_month = _MONTH_MAP[st_month]
            end_month = _MONTH_MAP[end_month]
            timestep = {v: k for k, v in _TIMESTEP_FREQUENCY_MAP.items()}[timestep]
        except AttributeError as e:
            raise ValueError("String does not match the expected format.") from e

    d = {
        "st_month": int(st_month),
        "st_day": int(st_day),
        "end_month": int(end_month),
        "end_day": int(end_day),
        "st_hour": int(st_hour),
        "end_hour": int(end_hour),
        "timestep": int(timestep),
        "is_leap_year": is_leap_year == "L",
    }

    return AnalysisPeriod.from_dict(d)

@bhom_analytics()
def describe_analysis_period(
    analysis_period: list[AnalysisPeriod],
    save_path: bool = False,
    include_timestep: bool = False,
) -> str:
    """Create a description of the given analysis period.

    Output from this method does not necessarily make a string that can be converted back to an AnalysisPeriod.
    To perform that, use `from ladybugtools_toolkit.ladybug_extension.analysisperiod import analysis_period_to_string, string_to_analysis_period`.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
        include_timestep (bool, optional):
            If True, include the timestep in the description.

    Returns:
        str:
            A description of the analysis period.
    """

    if isinstance(analysis_period, AnalysisPeriod):
        analysis_period = [analysis_period]

    # remove duplicates from list
    analysis_period = list(set(analysis_period))

    # check timestep of each analysis period is the same
    if len(set(ap.timestep for ap in analysis_period)) > 1:
        raise ValueError("All analysis periods must have the same timestep.")

    timestep = {
        1: "hour",
        2: "half-hour",
        3: "20 minutes",
        4: "15 minutes",
        5: "12 minutes",
        6: "10 minutes",
        10: "6 minutes",
        12: "5 minutes",
        15: "4 minutes",
        20: "3 minutes",
        30: "2 minutes",
        60: "minute",
    }

    if save_path:
        if len(analysis_period) != 1:
            raise ValueError("Only one analysis period can be used for a save path.")
        analysis_period = analysis_period[0]

        base_str = analysis_period_to_string(analysis_period, save_path)

        if include_timestep:
            return (
                f"{base_str}"
                f"{analysis_period.timestep:02}"
            )
        return base_str

    base_str = []
    for ap in analysis_period:
        base_str.apprnd(analysis_period_to_string(analysis_period))
    base_str = ", and ".join(base_str)

    if include_timestep:
        return f"{base_str}, every {timestep[analysis_period[0].timestep]}" #This assumes that all the analysis periods given have the same frequency as each other which may not be the case!

    return base_str

@bhom_analytics()
def do_analysis_periods_represent_entire_year(
    analysis_periods: list[AnalysisPeriod],
) -> bool:
    """Check a list of analysis periods to see if they represent an entire year.

    Args:
        analysis_periods (list[AnalysisPeriod]):
            A list of analysis periods.

    Returns:
        bool:
            True if the analysis periods represent an entire year, errors are
            raised otherwise.

    """
    if any(ap.end_hour < ap.st_hour for ap in analysis_periods):
        raise ValueError(
            "To combine time periods crossing midnight, AnalysisPeriod should "
            "be split into two parts - one for either side of midnight."
        )

    # Validation
    if any(ap.timestep != 1 for ap in analysis_periods):
        raise ValueError("All input analysis period timesteps must be hourly.")

    if any(
        ap.is_leap_year != analysis_periods[0].is_leap_year for ap in analysis_periods
    ):
        raise ValueError(
            "All input analysis periods must be either leap year, or not leap "
            "year. Mixed leapedness is not allowed."
        )

    target_datetimes = analysis_period_to_datetimes(AnalysisPeriod())
    actual_datetimes = (
        pd.concat(
            [analysis_period_to_datetimes(ap).to_series() for ap in analysis_periods]
        )
        .sort_index()
        .index
    )
    target_timesteps = 8784 if analysis_periods[0].is_leap_year else 8760
    actual_timesteps = sum(len(ap) for ap in analysis_periods)
    if actual_timesteps > target_timesteps:
        duplicates = actual_datetimes[actual_datetimes.duplicated()]
        raise ValueError(
            "The number of timesteps contained within the input analysis "
            f"periods is greater than {target_timesteps}. Duplicate timesteps "
            f"are {duplicates}"
        )
    if actual_timesteps < target_timesteps:
        # pylint: disable=E1125
        missing = (
            pd.DatetimeIndex(list(set(target_datetimes) - set(actual_datetimes)))
            .to_series()
            .sort_index()
            .index
        )
        # pylint: enable=E1125
        raise ValueError(
            "The number of timesteps contained within the input analysis "
            f"periods is less than {target_timesteps}. Missing timesteps "
            f"are {missing}"
        )

    return True
