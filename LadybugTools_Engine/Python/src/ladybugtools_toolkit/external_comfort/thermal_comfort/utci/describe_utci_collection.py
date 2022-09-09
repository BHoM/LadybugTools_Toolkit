from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


def describe_utci_collection(
    universal_thermal_climate_index: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = None,
) -> str:
    """Create a text summary of the given UTCI data collection.

    Returns:
        str: A text summary of the given UTCI data collection.
    """
    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    ap_description = describe_analysis_period(analysis_period)

    col = universal_thermal_climate_index.filter_by_analysis_period(analysis_period)
    series = to_series(col)

    total_number_of_hours = len(series)
    comfortable_hours = ((series >= 9) & (series <= 26)).sum()
    hot_hours = (series > 26).sum()
    cold_hours = (series < 9).sum()

    statements = [
        f'For {ap_description}, "No thermal stress" is expected for {comfortable_hours} out of a possible {total_number_of_hours} hours ({comfortable_hours/total_number_of_hours:0.1%}).',
        f'"Cold stress" is expected for {cold_hours} hours ({cold_hours/total_number_of_hours:0.1%}).',
        f'"Heat stress" is expected for {hot_hours} hours ({hot_hours/total_number_of_hours:0.1%}).',
    ]
    return " ".join(statements)
