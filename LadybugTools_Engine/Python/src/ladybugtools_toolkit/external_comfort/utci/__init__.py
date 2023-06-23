import calendar
import warnings
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybug_comfort.collection.utci import UTCI

from ...categorical import Categories, Category
from ...helpers import evaporative_cooling_effect
from ...ladybug_extension.analysis_period import analysis_period_to_boolean
from ...ladybug_extension.analysis_period import (
    describe_analysis_period as describe_analysis_period,
)
from ...ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ...ladybug_extension.epw import (
    seasonality_from_day_length,
    seasonality_from_month,
    seasonality_from_temperature,
)
from .process import met_rate_adjustment

UTCI_CATEGORIES = Categories(
    categories=[
        Category(
            name="Extreme cold stress",
            low_limit=-np.inf,
            high_limit=-40,
            color="#0D104B",
        ),
        Category(
            name="Very strong cold stress",
            low_limit=-40,
            high_limit=-27,
            color="#262972",
        ),
        Category(
            name="Strong cold stress",
            low_limit=-27,
            high_limit=-13,
            color="#3452A4",
        ),
        Category(
            name="Moderate cold stress",
            low_limit=-13,
            high_limit=0,
            color="#3C65AF",
        ),
        Category(
            name="Slight cold stress",
            low_limit=0,
            high_limit=9,
            color="#37BCED",
        ),
        Category(
            name="No thermal stress",
            low_limit=9,
            high_limit=26,
            color="#2EB349",
        ),
        Category(
            name="Moderate heat stress",
            low_limit=26,
            high_limit=32,
            color="#F38322",
        ),
        Category(
            name="Strong heat stress",
            low_limit=32,
            high_limit=38,
            color="#C31F25",
        ),
        Category(
            name="Very strong heat stress",
            low_limit=38,
            high_limit=46,
            color="#7F1416",
        ),
        Category(
            name="Extreme heat stress",
            low_limit=46,
            high_limit=np.inf,
            color="#580002",
        ),
    ]
)

UTCI_SIMPLIFIED_CATEGORIES = Categories(
    categories=[
        Category(
            name="Too cold (x < 9°C)",
            low_limit=-np.inf,
            high_limit=9,
            color="#3C65AF",
        ),
        Category(
            name="Comfortable (9°C ≤ x < 26°C)",
            low_limit=9,
            high_limit=26,
            color="#2EB349",
        ),
        Category(
            name="Too hot (x ≥ 26°C)",
            low_limit=26,
            high_limit=np.inf,
            color="#C31F25",
        ),
    ]
)


def summarise_utci_collections(
    utci_collections: List[HourlyContinuousCollection],
    categories: Categories = UTCI_CATEGORIES,
    mask: List[bool] = None,
    identifiers: Tuple[str] = None,
) -> pd.DataFrame:
    """Create a summary of a comparison between a "baseline" UTCI collection, and another.

    Args:
        utci_collections (List[HourlyContinuousCollection]):
            A list of UTCI collections to compare.
        categories (Categories, optional):
            A set of categories to use for the comparison.
        mask (List[bool], optional):
            An analysis period or list of boolean values to mask the collections by.
        identifiers (Tuple[str]):
            A tuple of identifiers for the baseline and comparable collections.

    Returns:
        str:
            A text summary of the differences between the given UTCI data collection.
    """

    # ensure each collection given it a UTCI collection
    if len(utci_collections) < 2:
        raise ValueError("At least two UTCI collections must be given to compare them.")
    if identifiers is None:
        identifiers = range(len(utci_collections))
    else:
        assert len(identifiers) == len(
            utci_collections
        ), "The identifiers given must be the same length as the collections given."

    return pd.concat(
        [
            categories.timeseries_summary_valuecounts(i, mask=mask)
            for i in utci_collections
        ],
        axis=1,
        names=identifiers,
    )


def utci_shade_benefit_categories(
    unshaded_utci: pd.Series,
    shaded_utci: pd.Series,
    comfort_limits: Tuple[float] = (9, 26),
) -> pd.Series:
    """Determine shade-gap analysis category, indicating where shade is not beneificial.

    Args:
        unshaded_utci (pd.Series):
            A series containing unshaded UTCI values.
        shaded_utci (pd.Series):
            A series containing shaded UTCI values.
        comfort_limits (Tuple[float], optional):
            The range within which "comfort" is achieved. Defaults to (9, 26).

    Returns:
        pd.Series:
            A catgorical series indicating shade-benefit.

    """

    if len(unshaded_utci) != len(shaded_utci):
        raise ValueError(
            f"Input sizes do not match ({len(unshaded_utci)} != {len(shaded_utci)})"
        )

    # get limits
    low, high = min(comfort_limits), max(comfort_limits)

    # get distance to comfort (degrees from edge of "comfortable")
    distance_from_comfort_unshaded = abs(
        np.where(
            unshaded_utci < low,
            unshaded_utci - low,
            np.where(unshaded_utci > high, unshaded_utci - high, 0),
        )
    )
    distance_from_comfort_shaded = abs(
        np.where(
            shaded_utci < low,
            shaded_utci - low,
            np.where(shaded_utci > high, shaded_utci - high, 0),
        )
    )

    # get boolean mask where comfortable
    comfortable_unshaded = unshaded_utci.between(low, high)
    comfortable_shaded = shaded_utci.between(low, high)

    # get masks for each category
    comfortable_without_shade = comfortable_unshaded
    comfortable_with_shade = ~comfortable_unshaded & comfortable_shaded
    shade_has_negative_impact = (
        distance_from_comfort_unshaded < distance_from_comfort_shaded
    )
    shade_has_positive_impact = (
        distance_from_comfort_unshaded > distance_from_comfort_shaded
    )

    # construct categorical series
    shade_categories = np.where(
        comfortable_without_shade,
        "Comfortable without shade",
        np.where(
            comfortable_with_shade,
            "Comfortable with shade",
            np.where(
                shade_has_negative_impact,
                "Shade is detrimental",
                np.where(shade_has_positive_impact, "Shade is beneficial", "Undefined"),
            ),
        ),
    )

    return pd.Series(shade_categories, index=unshaded_utci.index)


def distance_to_comfortable(
    values: Union[List[float], float],
    to_comfort_midpoint: bool = True,
    comfort_limits: Tuple[float] = None,
    absolute: bool = False,
) -> Union[List[float], float]:
    """
    Get the distance between the given value/s and the "comfortable" category.

    Args:
        values (Union[List[float], float]):
            A value or set of values representing UTCI temperature.
        to_comfort_midpoint (bool, optional):
            The point to which "distance" will be measured. Either the midpoint of the "No Thermal Stress" range, or the edge of the "No Thermal Stress" category. Defaults to True.
        comfort_limits (Tuple[float], optional):
            Bespoke comfort limits. Defaults to (9, 26).
        absolute: (bool, optional):
            Return values in absolute terms. Default is False.

    Returns:
        str:
            A text summary of the given UTCI data collection.

    """

    dtype = None
    if isinstance(values, pd.Series):
        idx = values.index
        name = values.name
        dtype = type(values)
    elif isinstance(values, pd.DataFrame):
        idx = values.index
        columns = values.columns
        dtype = type(values)

    if comfort_limits is None:
        comfort_limits = (9, 26)
    low = min(comfort_limits)
    high = max(comfort_limits)

    # convert to array
    values = np.array(values)

    comfort_midpoint = np.mean(comfort_limits)

    if to_comfort_midpoint:
        distance = np.where(
            values < comfort_midpoint,
            values - comfort_midpoint,
            comfort_midpoint - values,
        )
    else:
        distance = np.where(
            values < low, values - low, np.where(values > high, high - values, 0)
        )

    if absolute:
        distance = np.abs(distance)

    if dtype == pd.Series:
        return pd.Series(distance, index=idx, name=name)
    if dtype == pd.DataFrame:
        return pd.DataFrame(distance, index=idx, columns=columns)

    return distance


def categorise(
    values: Union[List[float], float],
    fmt: str = "category",
    simplified: bool = False,
    comfort_limits: Tuple[float] = (9, 26),
) -> Union[pd.Categorical, str]:
    """Convert a numeric values into their associated UTCI categories."""

    labels, bins = utci_comfort_categories(
        simplified=simplified,
        comfort_limits=comfort_limits,
        rtype=fmt,
    )

    if isinstance(values, (int, float)):
        return pd.cut([values], bins, labels=labels)[0]

    if isinstance(values, pd.DataFrame):
        df = pd.DataFrame(
            [
                pd.cut(series, bins, labels=labels).tolist()
                for _, series in values.iteritems()
            ]
        ).T
        df.index = values.index
        df.columns = values.columns
        return df

    return pd.cut(values, bins, labels=labels)


def feasible_utci_limits(
    epw: EPW, include_additional_moisture: float = 0, as_dataframe: bool = False
) -> Union[List[HourlyContinuousCollection], pd.DataFrame]:
    """Calculate the absolute min/max collections of UTCI based on possible shade, wind and moisture conditions.

    Args:
        epw (EPW):
            The EPW object for which limits will be calculated.
        as_dataframe (bool):
            Return the output as a dataframe with two columns, instread of two separate collections.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits. Default is 0, for no evaporative cooling.

    Returns:
        List[HourlyContinuousCollection]:
            The lowest UTCI and highest UTCI temperatures for each hour of the year.
    """
    mrt_unshaded = OutdoorSolarCal(
        epw.location,
        epw.direct_normal_radiation,
        epw.diffuse_horizontal_radiation,
        epw.horizontal_infrared_radiation_intensity,
        epw.dry_bulb_temperature,
    ).mean_radiant_temperature
    dbt_evap, rh_evap = np.array(
        [
            evaporative_cooling_effect(
                dry_bulb_temperature=_dbt,
                relative_humidity=_rh,
                evaporative_cooling_effectiveness=include_additional_moisture,
                atmospheric_pressure=_atm,
            )
            for _dbt, _rh, _atm in list(
                zip(
                    *[
                        epw.dry_bulb_temperature,
                        epw.relative_humidity,
                        epw.atmospheric_station_pressure,
                    ]
                )
            )
        ]
    ).T
    dbt_evap = epw.dry_bulb_temperature.get_aligned_collection(dbt_evap)
    rh_evap = epw.relative_humidity.get_aligned_collection(rh_evap)

    dbt_rh_options = (
        [[dbt_evap, rh_evap], [epw.dry_bulb_temperature, epw.relative_humidity]]
        if include_additional_moisture != 0
        else [[epw.dry_bulb_temperature, epw.relative_humidity]]
    )

    utcis = []
    utcis = []
    for _dbt, _rh in dbt_rh_options:
        for _ws in [
            epw.wind_speed,
            epw.wind_speed.get_aligned_collection(0),
            epw.wind_speed * 1.1,
        ]:
            for _mrt in [epw.dry_bulb_temperature, mrt_unshaded]:
                utcis.append(
                    UTCI(
                        air_temperature=_dbt,
                        rad_temperature=_mrt,
                        rel_humidity=_rh,
                        wind_speed=_ws,
                    ).universal_thermal_climate_index,
                )
    df = pd.concat([collection_to_series(i) for i in utcis], axis=1)
    min_utci = collection_from_series(
        df.min(axis=1).rename("Universal Thermal Climate Index (C)")
    )
    max_utci = collection_from_series(
        df.max(axis=1).rename("Universal Thermal Climate Index (C)")
    )

    if as_dataframe:
        return pd.concat(
            [
                collection_to_series(min_utci),
                collection_to_series(max_utci),
            ],
            axis=1,
            keys=["lowest", "highest"],
        )

    return min_utci, max_utci


def feasible_comfort_category(
    epw: EPW,
    analysis_periods: Union[List[AnalysisPeriod], AnalysisPeriod] = AnalysisPeriod(),
    density: bool = True,
    simplified: bool = False,
    comfort_limits: Tuple[float] = (9, 26),
    include_additional_moisture: bool = False,
    met_rate_adjustment_value: float = None,
) -> pd.DataFrame:
    """
    Based on the best/worst conditions that could be envisaged in an EPWs
    location, determine what the upper and lower bounds for UTCI values might
    be for each hour of the year.

    Args:
        epw (EPW):
            An EPW object
        analysis_periods (Union[List[AnalysisPeriod], AnalysisPeriod], optional):
            An analysis period or list of analysis periods to calculate the hours to be included in the output. Defaults to AnalysisPeriod().
        density (bool, optional):
            Return proportion of time rather than number of hours. Defaults to True.
        simplified (bool, optional):
            Simplify comfort categories to use below/within/upper instead of
            discrete UTCI categories. Defaults to False.
        comfort_limits (Tuple[float], optional):
            The lower and upper limits for the simplified comfort categories. Defaults to (9, 26).
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits. Defaults to 0 for no evap clg effect.
        met_rate_adjustment_value (float, optional):
            The value to adjust the metabolic rate by. Defaults to None which in turn defaults to the standard for UTCI of 2.3 MET.

    Returns:
        pd.DataFrame: _description_
    """

    # TODO - simplify this method to make use of the parts already completed in "describe_monthly"

    # check analysis periods are iterable
    if isinstance(analysis_periods, AnalysisPeriod):
        analysis_periods = [analysis_periods]
    elif any(not isinstance(ap, AnalysisPeriod) for ap in analysis_periods):
        raise TypeError("analysis_periods must be an iterable of AnalysisPeriods")

    hours = analysis_period_to_boolean(analysis_periods)

    for ap in analysis_periods:
        if (ap.st_month != 1) or (ap.end_month != 12):
            raise ValueError("Analysis periods must be for the whole year.")

    cats, _ = utci_comfort_categories(
        simplified=simplified,
        comfort_limits=comfort_limits,
    )
    # calculate lower and upper bounds
    df = feasible_utci_limits(
        epw, as_dataframe=True, include_additional_moisture=include_additional_moisture
    )
    if met_rate_adjustment_value is not None:
        df["lowest"] = met_rate_adjustment(
            collection_from_series(
                df["lowest"].rename("Universal Thermal Climate Index (C)")
            ),
            met_rate_adjustment_value,
        ).values
        df["highest"] = met_rate_adjustment(
            collection_from_series(
                df["highest"].rename("Universal Thermal Climate Index (C)")
            ),
            met_rate_adjustment_value,
        ).values

    # filter by hours
    df_filtered = df.loc[hours]

    # categorise
    df_cat = categorise(
        df_filtered, simplified=simplified, comfort_limits=comfort_limits
    )

    # join categories and get low/high lims
    temp = pd.concat(
        [
            df_cat.groupby(df_cat.index.month)
            .lowest.value_counts(normalize=density)
            .unstack()
            .reindex(cats, axis=1)
            .fillna(0),
            df_cat.groupby(df_cat.index.month)
            .highest.value_counts(normalize=density)
            .unstack()
            .reindex(cats, axis=1)
            .fillna(0),
        ],
        axis=1,
    )
    columns = pd.MultiIndex.from_product([cats, ["lowest", "highest"]])

    temp = pd.concat(
        [
            temp.groupby(temp.columns, axis=1).min(),
            temp.groupby(temp.columns, axis=1).max(),
        ],
        axis=1,
        keys=["lowest", "highest"],
    ).reorder_levels(order=[1, 0], axis=1)[columns]
    temp.index = [calendar.month_abbr[i] for i in temp.index]

    return temp


def feasible_comfort_temporal(
    epw: EPW,
    st_hour: float = 0,
    end_hour: float = 23,
    seasonality: Union[Callable, str] = "monthly",
    comfort_limits: Tuple[float] = (9, 26),
    include_additional_moisture: float = 0,
) -> pd.DataFrame:
    """
    Based on the min/max feasible proportion of time where comfort can be
    achieved, get the temporal probabilities that this may happen for the
    given summarisation method.

    Args:
        epw (EPW):
            An EPW object.
        st_hour (float, optional):
            The start hour for any time-based filtering to apply. Defaults to 0.
        end_hour (float, optional):
            The end-hour for any time-based filtering to apply. Defaults to 23.
        seasonality (Callable, optional):
            How to present results in any annual-seasonal summarisation. Defaults to "monthly".
        comfort_limits (List[float], optional):
            What is considerred the upper and lower limits of "comfort". Defaults to [9, 26] per UTCI standard.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits. Defaults to 0 for no evap clg.

    Returns:
        pd.DataFrame:
            A summary table.
    """

    seasonality_methods = [
        None,
        seasonality_from_day_length,
        seasonality_from_month,
        seasonality_from_temperature,
        "monthly",
    ]
    if seasonality not in seasonality_methods:
        raise ValueError(f"seasonality must be one of {seasonality_methods}")

    min_utci, max_utci = feasible_utci_limits(
        epw, include_additional_moisture=include_additional_moisture
    )

    utci_range = pd.concat(
        [collection_to_series(min_utci), collection_to_series(max_utci)], axis=1
    ).agg(["min", "max"], axis=1)

    analysis_period = AnalysisPeriod(st_hour=st_hour, end_hour=end_hour)
    ap_bool = analysis_period_to_boolean(analysis_period)
    ap_description = describe_analysis_period(analysis_period)

    low_limit = min(comfort_limits)
    high_limit = max(comfort_limits)

    temp = utci_range[ap_bool]

    if seasonality == "monthly":
        temp = (temp >= low_limit) & (temp <= high_limit)
        temp = (
            temp.groupby(temp.index.month).sum()
            / temp.groupby(temp.index.month).count()
        )
        temp.columns = [
            "Minimum comfortable time [0-1]",
            "Maximum comfortable time [0-1]",
        ]
        temp.index = [calendar.month_abbr[i] for i in temp.index]
        return temp

    if seasonality is None:
        temp = (
            (((temp >= low_limit) & (temp <= high_limit)).sum() / temp.count())
            .sort_values()
            .to_frame()
        )
        temp.index = [
            "Minimum comfortable time [0-1]",
            "Maximum comfortable time [0-1]",
        ]
        temp.columns = [ap_description]
        return temp.T

    # pylint disable=comparison-with-callable
    if seasonality == seasonality_from_month:
        seasons = seasonality_from_month(epw, annotate=True)[ap_bool]
        keys = seasons.unique()
        temp = ((temp >= low_limit) & (temp <= high_limit)).groupby(
            seasons
        ).sum() / temp.groupby(seasons).count()
        temp = temp.agg(["min", "max"], axis=1)
        temp.columns = [
            "Minimum comfortable time [0-1]",
            "Maximum comfortable time [0-1]",
        ]
        temp = temp.T[keys].T
        temp.index = [
            f"{i} between {st_hour:02d}:00 and {end_hour:02d}:00" for i in temp.index
        ]
        return temp

    if seasonality == seasonality_from_day_length:
        seasons = seasonality_from_day_length(epw, annotate=True)[ap_bool]
        keys = seasons.unique()
        temp = ((temp >= low_limit) & (temp <= high_limit)).groupby(
            seasons
        ).sum() / temp.groupby(seasons).count()
        temp = temp.agg(["min", "max"], axis=1)
        temp.columns = [
            "Minimum comfortable time [0-1]",
            "Maximum comfortable time [0-1]",
        ]
        temp = temp.T[keys].T
        temp.index = [
            f"{i} between {st_hour:02d}:00 and {end_hour:02d}:00" for i in temp.index
        ]
        return temp

    if seasonality == seasonality_from_temperature:
        seasons = seasonality_from_temperature(epw, annotate=True)[ap_bool]
        keys = seasons.unique()
        temp = ((temp >= low_limit) & (temp <= high_limit)).groupby(
            seasons
        ).sum() / temp.groupby(seasons).count()
        temp = temp.agg(["min", "max"], axis=1)
        temp.columns = [
            "Minimum comfortable time [0-1]",
            "Maximum comfortable time [0-1]",
        ]
        temp = temp.T[keys].T
        temp.index = [
            f"{i} between {st_hour:02d}:00 and {end_hour:02d}:00" for i in temp.index
        ]
        return temp
    # pylint enable=comparison-with-callable

    raise ValueError("How did you get here?")


def utci_comfort_categories(
    simplified: bool = False,
    comfort_limits: Tuple[float] = (9, 26),
    rtype: str = "category",
) -> List[List[Any]]:
    """Create a list of UTCI comfort categories based on given configuration.

    Args:
        simplified (bool, optional):
            Return simplified categories. Defaults to False.
        comfort_limits (Tuple[float], optional):
            Modify simplified categories. Defaults to (9, 26).
        rtype (str, optional):
            Return type - category or color. Defaults to "category".

    Returns:
        List[List[Any]]:
            A list of UTCI comfort categories - in the form [[category_0, category_n], [bound_0, ..., bound_n+1]]
    """

    rtypes = ["category", "color"]
    if rtype not in rtypes:
        raise ValueError(f"rtype must be one of {rtypes}")

    if rtype == "color":
        if simplified:
            labels = ["#3C65AF", "#2EB349", "#C31F25"]
        else:
            labels = UTCI_CATEGORIES.colors
    else:
        if simplified:
            labels = [
                f"Too cold (<{min(comfort_limits)}°C)",
                f"Comfortable ({min(comfort_limits)}°C ≤ x ≤ {max(comfort_limits)}°C)",
                f"Too hot (>{max(comfort_limits)}°C)",
            ]
        else:
            labels = UTCI_CATEGORIES.names

    if simplified:
        bounds = [-np.inf, min(comfort_limits), max(comfort_limits), np.inf]
    else:
        bounds = UTCI_CATEGORIES._bin_edges()

    return (
        labels,
        bounds,
    )


def month_time_binned_table(
    utci_data: Union[pd.Series, HourlyContinuousCollection],
    month_bins: Tuple[List[int]],
    hour_bins: Tuple[List[int]],
    simplified: bool = False,
    comfort_limits: Tuple[float] = (9, 26),
    color_result: bool = False,
    time_labels: List[str] = None,
    month_labels: List[str] = None,
) -> pd.DataFrame:
    """Create a table with monthly time binned UTCI data.

    Args:
        utci_data (Union[pd.Series, HourlyContinuousCollection]):
            A collection of UTCI values.
        month_bins (Tuple[List[int]]):
            A list of lists of months to group data into.
        hour_bins (Tuple[List[int]]):
            A list of lists of hours to group data into.
        comfort_limits (Tuple[float], optional):
            The comfortable limits within which "comfort" is achieved. Defaults to (9, 26).
        simplified (bool, optional):
            Return simplified categories. Defaults to False.
        color_result (bool, optional):
            Return a color-coded table. Defaults to False.
        time_labels (List[str], optional):
            A list of labels for the time bins. Defaults to None.
        month_labels (List[str], optional):
            A list of labels for the month bins. Defaults to None.

    Returns:
        pd.DataFrame:
            A table with monthly time binned UTCI data.
    """

    # check the utci_data is either a pd.Series or a HourlyContinuousCollection
    if not isinstance(utci_data, (pd.Series, HourlyContinuousCollection)):
        raise TypeError(
            "utci_data must be either a pandas.Series or a ladybug.datacollection.HourlyContinuousCollection"
        )

    if isinstance(utci_data, HourlyContinuousCollection):
        utci_data = collection_to_series(utci_data)

    # check for continuity of time periods, and overlaps overnight/year
    flat_hours = [item for sublist in hour_bins for item in sublist]
    flat_months = [item for sublist in month_bins for item in sublist]

    if (max(flat_hours) != 23) or min(flat_hours) != 0:
        raise ValueError("hour_bins hours must be in the range 0-23")
    if (max(flat_months) != 12) or min(flat_months) != 1:
        raise ValueError("month_bins hours must be in the range 1-12")
    if (set(flat_hours) != set(list(range(24)))) or (len(set(flat_hours)) != 24):
        raise ValueError("Input hour_bins does not contain all hours of the day")
    if (set(flat_months) != set(list(range(1, 13, 1)))) or (
        len(set(flat_months)) != 12
    ):
        raise ValueError("Input month_bins does not contain all months of the year")

    # create index/column labels
    if month_labels:
        if len(month_labels) != len(month_bins):
            raise ValueError("month_labels must be the same length as month_bins")
        row_labels = month_labels
    else:
        row_labels = [
            f"{calendar.month_abbr[i[0]]} to {calendar.month_abbr[i[-1]]}"
            for i in month_bins
        ]
    if time_labels:
        if len(time_labels) != len(hour_bins):
            raise ValueError("time_labels must be the same length as hour_bins")
        col_labels = time_labels
    else:
        col_labels = [f"{i[0]:02d}:00 ≤ x < {i[-1] + 1:02d}:00" for i in hour_bins]

    # create indexing bins
    values = []
    for months in month_bins:
        month_mask = utci_data.index.month.isin(months)
        inner_values = []
        for hours in hour_bins:
            mask = utci_data.index.hour.isin(hours) & month_mask
            avg = utci_data.loc[mask].mean()
            inner_values.append(avg)
        values.append(inner_values)
    df = pd.DataFrame(values, index=row_labels, columns=col_labels).T

    if color_result:
        warnings.warn(
            'The value returned by this method when "color_result" is applied is not a pd.Dataframe object. To get the dataframe use ".data".'
        )

        def _highlight(val):
            return f'color:black;background-color:{categorise(val, comfort_limits=comfort_limits, simplified=simplified, fmt="color")}'

        return df.style.applymap(_highlight)

    return df
