from typing import Tuple

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_analysis_period,
)
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series


from python_toolkit.bhom.analytics import analytics


@analytics
def compare_utci_collections(
    baseline: HourlyContinuousCollection,
    comparable: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = None,
    identifiers: Tuple[str] = ("UTCI collection 1", "UTCI collection 2"),
) -> str:
    """Create a summary of a comparison between a "baseline" UTCI collection, and another.

    Returns:
        str: A text summary of the differences between the given UTCI data collection.
    """
    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    if len(identifiers) != 2:
        raise ValueError('The number of "identifiers" must be equal to 2.')

    if len(baseline) != len(comparable):
        raise ValueError(
            "The collections given are not comparable as they are not the same length."
        )

    ap_description = describe_analysis_period(analysis_period)

    col_baseline = baseline.filter_by_analysis_period(analysis_period)
    series_baseline = to_series(col_baseline)
    col_comparable = comparable.filter_by_analysis_period(analysis_period)
    series_comparable = to_series(col_comparable)

    total_number_of_hours = len(col_baseline)
    comfortable_hours_baseline = (
        (series_baseline >= 9) & (series_baseline <= 26)
    ).sum()
    hot_hours_baseline = (series_baseline > 26).sum()
    cold_hours_baseline = (series_baseline < 9).sum()

    comfortable_hours_comparable = (
        (series_comparable >= 9) & (series_comparable <= 26)
    ).sum()
    hot_hours_comparable = (series_comparable > 26).sum()
    cold_hours_comparable = (series_comparable < 9).sum()

    comfortable_hours_difference = (
        comfortable_hours_baseline - comfortable_hours_comparable
    )
    comfortable_hours_worse = True if comfortable_hours_difference > 0 else False
    hot_hours_difference = hot_hours_baseline - hot_hours_comparable
    hot_hours_worse = False if hot_hours_difference > 0 else True
    cold_hours_difference = cold_hours_baseline - cold_hours_comparable
    cold_hours_worse = False if cold_hours_difference > 0 else True

    # print("total_number_of_hours", total_number_of_hours)
    # print("comfortable_hours_baseline", comfortable_hours_baseline)
    # print("hot_hours_baseline", hot_hours_baseline)
    # print("cold_hours_baseline", cold_hours_baseline)
    # print("comfortable_hours_comparable", comfortable_hours_comparable)
    # print("hot_hours_comparable", hot_hours_comparable)
    # print("cold_hours_comparable", cold_hours_comparable)
    # print("comfortable_hours_difference", comfortable_hours_difference)
    # print("comfortable_hours_worse", comfortable_hours_worse)
    # print("hot_hours_difference", hot_hours_difference)
    # print("hot_hours_worse", hot_hours_worse)
    # print("cold_hours_difference", cold_hours_difference)
    # print("cold_hours_worse", cold_hours_worse)

    statements = [
        f'For {ap_description}, "{identifiers[1]}" is generally {"less" if comfortable_hours_worse else "more"} thermally comfortable than "{identifiers[0]}" (with a {abs(comfortable_hours_difference / comfortable_hours_baseline):0.1%} {"reduction" if comfortable_hours_worse else "increase"} in number of hours experiencing "no thermal stress").',
        f'"{identifiers[1]}" demonstrates a {abs(hot_hours_difference / hot_hours_baseline):0.1%} {"increase" if hot_hours_worse else "decrease"} in number of hours experiencing some form of "heat stress" from "{identifiers[0]}"',
        f'"{identifiers[1]}" demonstrates a {abs(cold_hours_difference / cold_hours_baseline):0.1%} {"increase" if cold_hours_worse else "decrease"} in number of hours experiencing some form of "cold stress" from "{identifiers[0]}".',
    ]

    return " ".join(statements)
