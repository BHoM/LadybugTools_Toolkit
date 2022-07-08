import calendar

from ladybug.analysisperiod import AnalysisPeriod


def describe(analysis_period: AnalysisPeriod, save_path: bool = False) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod): A Ladybug analysis period.
        save_path (str, optional): If True, create a path-safe string from the analysis period.

    Returns:
        str: A description of the analysis period.
    """

    if save_path:
        return f"{analysis_period.st_month:02}{analysis_period.st_day:02}_{analysis_period.end_month:02}{analysis_period.end_day:02}_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"
    else:
        return f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to {calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between {analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:00"
