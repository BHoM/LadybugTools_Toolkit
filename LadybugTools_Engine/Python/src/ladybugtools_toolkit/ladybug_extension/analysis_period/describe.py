import calendar

from ladybug.analysisperiod import AnalysisPeriod


from ladybugtools_toolkit import analytics


@analytics
def describe(
    analysis_period: AnalysisPeriod,
    save_path: bool = False,
    include_timestep: bool = True,
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod): A Ladybug analysis period.
        save_path (bool, optional): If True, create a path-safe string from the analysis period.
        include_timestep (bool, optional): If True, include the timestep in the description.

    Returns:
        str: A description of the analysis period.
    """

    if save_path:
        return f"{analysis_period.st_month:02}{analysis_period.st_day:02}_{analysis_period.end_month:02}{analysis_period.end_day:02}_{analysis_period.st_hour:02}_{(analysis_period.end_hour + 1):02}"
    else:
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

        output = f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to {calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between {analysis_period.st_hour:02}:00 and {(analysis_period.end_hour + 1):02}:00."

        if include_timestep:
            return output[:-1] + f", every {timestep[analysis_period.timestep]}"

        return output
