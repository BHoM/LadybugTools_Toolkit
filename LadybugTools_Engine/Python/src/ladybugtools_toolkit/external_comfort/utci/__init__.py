import calendar
import warnings
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import (
    UniversalThermalClimateIndex as LB_UniversalThermalClimateIndex,
)
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybug_comfort.collection.utci import UTCI

from ...categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical
from ...helpers import evaporative_cooling_effect
from ...ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)


def compare_monthly_utci(
    utci_collections: List[HourlyContinuousCollection],
    utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES,
    identifiers: Tuple[str] = None,
    density: bool = False,
    simplify: bool = False,
) -> pd.DataFrame:
    """Create a summary of a comparison between a "baseline" UTCI collection, and another.

    Args:
        utci_collections (List[HourlyContinuousCollection]):
            A list of UTCI collections to compare.
        utci_categories (Categories, optional):
            A set of categories to use for the comparison.
        identifiers (Tuple[str]):
            A tuple of identifiers for the baseline and comparable collections.
        density (bool, optional):
            Return the proportion of time rather than the number of hours. Defaults to False.
        simplify (bool, optional):
            Simplify the summary table to use the "ComfortClass" of the UTCI categories (where these attribtues have been added to the object). Defaults to False.

    Returns:
        str:
            A text summary of the differences between the given UTCI data collection.
    """

    # ensure each collection given it a UTCI collection
    if len(utci_collections) < 2:
        raise ValueError("At least two UTCI collections must be given to compare them.")
    # check each collection is a utci collection

    if any(
        not isinstance(i.header.data_type, LB_UniversalThermalClimateIndex)
        for i in utci_collections
    ):
        raise ValueError("Input collection is not a UTCI collection.")

    if identifiers is None:
        identifiers = range(len(utci_collections))
    else:
        assert len(identifiers) == len(
            utci_collections
        ), "The identifiers given must be the same length as the collections given."

    dd = []
    for i in utci_collections:
        _df = utci_categories.timeseries_summary_monthly(
            collection_to_series(i), density=density
        )
        _df.columns = [utci_categories.interval_from_bin_name(i) for i in _df.columns]
        dd.append(_df)
    df = pd.concat(dd, axis=1, keys=identifiers)
    df = df.reorder_levels([1, 0], axis=1).sort_index(axis=1)
    df = df.rename(columns=utci_categories._interval_bin_name)

    if simplify:
        try:
            if len(utci_categories.comfort_classes) != len(
                utci_categories.interval_index
            ):
                raise ValueError(
                    "Monkey-patched comfort_class attributes are not the same length as the categories within this Category object."
                )
            if set(["Too cold", "Comfortable", "Too hot"]) != set(
                [i.value for i in utci_categories.comfort_classes]
            ):
                raise ValueError(
                    'Monkey-patched comfort_class attributes should contain values of either "Too cold", "Comfortable" or "Too hot".'
                )

            ddd = {
                i: cc
                for cc, i in list(
                    zip(*[utci_categories.comfort_classes, utci_categories.bin_names])
                )
            }
            cc_header = []
            for col, _ in df.iteritems():
                cc_header.append(ddd[col[0]].value)

            df.columns = pd.MultiIndex.from_arrays(
                [
                    df.columns.get_level_values(0),
                    df.columns.get_level_values(1),
                    cc_header,
                ],
                names=["Category", "Identifier", "ComfortClass"],
            )
            df = df.groupby(
                [
                    df.columns.get_level_values("ComfortClass"),
                    df.columns.get_level_values("Identifier"),
                ],
                axis=1,
            ).sum()
            df = df.reindex(
                columns=df.columns.reindex(
                    ["Too cold", "Comfortable", "Too hot"], level=0
                )[0]
            )
            return df

        except AttributeError as exc:
            raise AttributeError(
                'No ComfortClass found on given "utci_categories", you cannot use these categories for a simplified summary until you add these.'
            ) from exc

    return df


def shade_benefit_category(
    unshaded_utci: Union[HourlyContinuousCollection, pd.Series],
    shaded_utci: Union[HourlyContinuousCollection, pd.Series],
    comfort_limits: Tuple[float] = (9, 26),
) -> pd.Series:
    """Determine shade-gap analysis category, indicating where shade is not beneificial.

    Args:
        unshaded_utci (Union[HourlyContinuousCollection, pd.Series]):
            A dataset containing unshaded UTCI values.
        shaded_utci (Union[HourlyContinuousCollection, pd.Series]):
            A dataset containing shaded UTCI values.
        comfort_limits (Tuple[float], optional):
            The range within which "comfort" is achieved. Defaults to (9, 26).

    Returns:
        pd.Series:
            A catgorical series indicating shade-benefit.

    """

    # convert to series if not already
    if isinstance(unshaded_utci, HourlyContinuousCollection):
        unshaded_utci = collection_to_series(unshaded_utci)
    if isinstance(shaded_utci, HourlyContinuousCollection):
        shaded_utci = collection_to_series(shaded_utci)

    if len(unshaded_utci) != len(shaded_utci):
        raise ValueError(
            f"Input sizes do not match ({len(unshaded_utci)} != {len(shaded_utci)})"
        )

    if sum(unshaded_utci == shaded_utci) == len(unshaded_utci):
        raise ValueError("Input series are identical.")

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
    values: Any,
    comfort_thresholds: Tuple[float] = (9, 26),
    distance_to_comfort_band_centroid: bool = True,
) -> Union[List[float], float]:
    """
    Get the distance between the given value/s and the "comfortable" category.

    Args:
        values (Any):
            A value or set of values representing UTCI temperature.
        comfort_thresholds (List[float], optional):
            The comfortable band of UTCI temperatures. Defaults to [9, 26].
        distance_to_comfort_band_centroid (bool, optional):
            If True, the distance to the centroid of the comfort band is plotted. If False, the
            distance to the edge of the comfort band is plotted. Defaults to False.

    Returns:
        str:
            A text summary of the given UTCI data collection.

    """

    if len(comfort_thresholds) != 2:
        raise ValueError("comfort_thresholds must be a list of length 2.")

    if len(set(comfort_thresholds)) != 2:
        raise ValueError("comfort_thresholds must contain two unique values.")

    if comfort_thresholds[0] > comfort_thresholds[1]:
        warnings.warn("comfort_thresholds are not increasing. Swapping the values.")

    low_limit = min(comfort_thresholds)
    high_limit = max(comfort_thresholds)
    midpoint = np.mean(comfort_thresholds)

    vals = np.array(values.values)
    if not distance_to_comfort_band_centroid:
        vals = np.where(
            vals < low_limit,
            vals - low_limit,
            np.where(vals > high_limit, vals - high_limit, 0),
        )
    else:
        vals = np.where(vals < midpoint, -(midpoint - vals), vals - midpoint)

    name = (
        f'Distance from "comfortable" [between {low_limit}°C and {high_limit}°C UTCI] (C)'
        if not distance_to_comfort_band_centroid
        else f'Distance from "comfortable" range midpoint [at {midpoint:0.1f}°C between {low_limit}°C and {high_limit}°C UTCI] (C)'
    )

    if isinstance(values, pd.Series):
        return pd.Series(vals, index=values.index, name=name)

    if isinstance(values, pd.DataFrame):
        return pd.DataFrame(vals, index=values.index, columns=values.columns)

    if isinstance(values, HourlyContinuousCollection):
        if not isinstance(values.header.data_type, LB_UniversalThermalClimateIndex):
            raise ValueError("Input collection is not a UTCI collection.")
        return collection_from_series(
            pd.Series(
                vals,
                index=collection_to_series(values).index,
                name=name,
            )
        )

    return vals


def feasible_utci_limits(
    epw: EPW, include_additional_moisture: float = 0, as_dataframe: bool = False
) -> Union[Tuple[HourlyContinuousCollection], pd.DataFrame]:
    """Calculate the absolute min/max collections of UTCI based on possible shade, wind and moisture conditions.

    Args:
        epw (EPW):
            The EPW object for which limits will be calculated.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits. Default is 0, for no evaporative cooling.
        as_dataframe (bool):
            Return the output as a dataframe with two columns, instread of two separate collections.

    Returns:
        Union[Tuple[HourlyContinuousCollection], pd.DataFrame]:
            The lowest possible UTCI and highest UTCI temperatures for each hour of the year.
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


def feasible_utci_category_limits(
    epw: EPW,
    include_additional_moisture: float = 0,
    utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES,
    density: bool = False,
    simplify: bool = False,
    mask: List[bool] = None,
):
    """Calculate the upper and lower proportional limits of UTCI categories based on possible shade, wind and moisture conditions.

    Args:
        epw (EPW):
            The EPW object for which limits will be calculated.
        include_additional_moisture (float, optional):
            Include the effect of evaporative cooling on the UTCI limits. Default is 0, for no evaporative cooling.
        utci_categories (Categorical, optional):
            A set of categories to use for the comparison.
        density (bool, optional):
            Return the proportion of time rather than the number of hours. Defaults to False.
        simplify (bool, optional):
            Simplify the summary table to use the "ComfortClass" of the UTCI categories (where these attribtues have been added to the object). Defaults to False.
        mask (List[bool], optional):
            A list of booleans to mask the data. Defaults to None.

    Returns:
        pd.DataFrame:
            A table with monthly time binned UTCI data.
    """

    lims = feasible_utci_limits(
        epw, include_additional_moisture=include_additional_moisture, as_dataframe=True
    )

    if mask is not None:
        lims = lims[mask]
        if lims.index.month.nunique() != 12:
            raise ValueError("Masked data include at least one value per month.")

    df = pd.concat(
        [
            utci_categories.timeseries_summary_monthly(lims.lowest, density=density),
            utci_categories.timeseries_summary_monthly(lims.highest, density=density),
        ],
        axis=1,
        keys=lims.columns,
    )
    df = (
        df.sort_index(axis=1)
        .reorder_levels([1, 0], axis=1)
        .sort_index(axis=1, ascending=[True, False])
    )
    df = df.rename(columns=utci_categories._interval_bin_name)

    if simplify:
        try:
            if len(utci_categories.comfort_classes) != len(
                utci_categories.interval_index
            ):
                raise ValueError(
                    "Monkey-patched comfort_class attributes are not the same length as the categories within this Category object."
                )
            if set(["Too cold", "Comfortable", "Too hot"]) != set(
                i.value for i in utci_categories.comfort_classes
            ):
                raise ValueError(
                    'Monkey-patched comfort_class attributes should contain values of either "Too cold", "Comfortable" or "Too hot".'
                )

            ddd = {
                i: cc
                for cc, i in list(
                    zip(*[utci_categories.comfort_classes, utci_categories.bin_names])
                )
            }
            cc_header = []
            for col, _ in df.iteritems():
                cc_header.append(ddd[col[0]].value)

            df.columns = pd.MultiIndex.from_arrays(
                [
                    df.columns.get_level_values(0),
                    df.columns.get_level_values(1),
                    cc_header,
                ],
                names=["Category", "Identifier", "ComfortClass"],
            )
            df = df.groupby(
                [
                    df.columns.get_level_values("ComfortClass"),
                    df.columns.get_level_values("Identifier"),
                ],
                axis=1,
            ).sum()
            df = df.sort_index(axis=1, ascending=[True, False])
            df = df.reindex(
                columns=df.columns.reindex(
                    ["Too cold", "Comfortable", "Too hot"], level=0
                )[0]
            )

            return df

        except AttributeError as exc:
            raise AttributeError(
                'No ComfortClass found on given "utci_categories", you cannot use these categories for a simplified summary until you add these.'
            ) from exc

    return df


# def month_time_binned_table(
#     utci_data: Union[pd.Series, HourlyContinuousCollection],
#     month_bins: Tuple[List[int]],
#     hour_bins: Tuple[List[int]],
#     simplified: bool = False,
#     comfort_limits: Tuple[float] = (9, 26),
#     color_result: bool = False,
#     time_labels: List[str] = None,
#     month_labels: List[str] = None,
# ) -> pd.DataFrame:
#     """Create a table with monthly time binned UTCI data.

#     Args:
#         utci_data (Union[pd.Series, HourlyContinuousCollection]):
#             A collection of UTCI values.
#         month_bins (Tuple[List[int]]):
#             A list of lists of months to group data into.
#         hour_bins (Tuple[List[int]]):
#             A list of lists of hours to group data into.
#         comfort_limits (Tuple[float], optional):
#             The comfortable limits within which "comfort" is achieved. Defaults to (9, 26).
#         simplified (bool, optional):
#             Return simplified categories. Defaults to False.
#         color_result (bool, optional):
#             Return a color-coded table. Defaults to False.
#         time_labels (List[str], optional):
#             A list of labels for the time bins. Defaults to None.
#         month_labels (List[str], optional):
#             A list of labels for the month bins. Defaults to None.

#     Returns:
#         pd.DataFrame:
#             A table with monthly time binned UTCI data.
#     """

#     # check the utci_data is either a pd.Series or a HourlyContinuousCollection
#     if not isinstance(utci_data, (pd.Series, HourlyContinuousCollection)):
#         raise TypeError(
#             "utci_data must be either a pandas.Series or a ladybug.datacollection.HourlyContinuousCollection"
#         )

#     if isinstance(utci_data, HourlyContinuousCollection):
#         utci_data = collection_to_series(utci_data)

#     # check for continuity of time periods, and overlaps overnight/year
#     flat_hours = [item for sublist in hour_bins for item in sublist]
#     flat_months = [item for sublist in month_bins for item in sublist]

#     if (max(flat_hours) != 23) or min(flat_hours) != 0:
#         raise ValueError("hour_bins hours must be in the range 0-23")
#     if (max(flat_months) != 12) or min(flat_months) != 1:
#         raise ValueError("month_bins hours must be in the range 1-12")
#     if (set(flat_hours) != set(list(range(24)))) or (len(set(flat_hours)) != 24):
#         raise ValueError("Input hour_bins does not contain all hours of the day")
#     if (set(flat_months) != set(list(range(1, 13, 1)))) or (
#         len(set(flat_months)) != 12
#     ):
#         raise ValueError("Input month_bins does not contain all months of the year")

#     # create index/column labels
#     if month_labels:
#         if len(month_labels) != len(month_bins):
#             raise ValueError("month_labels must be the same length as month_bins")
#         row_labels = month_labels
#     else:
#         row_labels = [
#             f"{calendar.month_abbr[i[0]]} to {calendar.month_abbr[i[-1]]}"
#             for i in month_bins
#         ]
#     if time_labels:
#         if len(time_labels) != len(hour_bins):
#             raise ValueError("time_labels must be the same length as hour_bins")
#         col_labels = time_labels
#     else:
#         col_labels = [f"{i[0]:02d}:00 ≤ x < {i[-1] + 1:02d}:00" for i in hour_bins]

#     # create indexing bins
#     values = []
#     for months in month_bins:
#         month_mask = utci_data.index.month.isin(months)
#         inner_values = []
#         for hours in hour_bins:
#             mask = utci_data.index.hour.isin(hours) & month_mask
#             avg = utci_data.loc[mask].mean()
#             inner_values.append(avg)
#         values.append(inner_values)
#     df = pd.DataFrame(values, index=row_labels, columns=col_labels).T

#     if color_result:
#         warnings.warn(
#             'The value returned by this method when "color_result" is applied is not a pd.Dataframe object. To get the dataframe use ".data".'
#         )

#         def _highlight(val):
#             return f'color:black;background-color:{categorise(val, comfort_limits=comfort_limits, simplified=simplified, fmt="color")}'

#         return df.style.applymap(_highlight)

#     return df
