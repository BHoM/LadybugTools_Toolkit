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
from scipy.interpolate import interp1d, interp2d

from ...categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical
from ...helpers import evaporative_cooling_effect, month_hour_binned_series
from ...ladybug_extension.datacollection import (
    collection_from_series,
    collection_to_series,
)
from ...plot.utilities import contrasting_color


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
    utci_values: Any,
    comfort_thresholds: Tuple[float] = (9, 26),
    distance_to_comfort_band_centroid: bool = True,
) -> Union[List[float], float]:
    """
    Get the distance between the given value/s and the "comfortable" category.

    Args:
        utci_values (Any):
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

    vals = np.array(utci_values.values)
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

    if isinstance(utci_values, pd.Series):
        return pd.Series(vals, index=utci_values.index, name=name)

    if isinstance(utci_values, pd.DataFrame):
        return pd.DataFrame(vals, index=utci_values.index, columns=utci_values.columns)

    if isinstance(utci_values, HourlyContinuousCollection):
        if not isinstance(
            utci_values.header.data_type, LB_UniversalThermalClimateIndex
        ):
            raise ValueError("Input collection is not a UTCI collection.")
        return collection_from_series(
            pd.Series(
                vals,
                index=collection_to_series(utci_values).index,
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


def month_hour_binned(
    utci_data: Union[pd.Series, HourlyContinuousCollection],
    month_bins: Tuple[Tuple[int]] = None,
    hour_bins: Tuple[Tuple[int]] = None,
    utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES,
    color_result: bool = False,
    month_labels: Tuple[str] = None,
    hour_labels: Tuple[str] = None,
    agg: str = "mean",
    **kwargs,
) -> pd.DataFrame:
    """Create a table with monthly hour binned UTCI data.

    Args:
        utci_data (Union[pd.Series, HourlyContinuousCollection]):
            A collection of UTCI values.
        month_bins (Tuple[List[int]]):
            A list of lists of months to group data into.
        hour_bins (Tuple[List[int]]):
            A list of lists of hours to group data into.
        utci_categories (Categorical, optional):
            A set of UTCI categories to use for the comparison.
        color_result (bool, optional):
            Return a color-coded table. Defaults to False.
        month_labels (List[str], optional):
            A list of labels for the month bins. Defaults to None.
        hour_labels (List[str], optional):
            A list of labels for the hour bins. Defaults to None.
        agg (str, optional):
            The aggregation method to use for the data within each bin. Defaults to "mean".

    Keyword Args:
        contrast_text (bool, optional):
            If True, the text color will be chosen to contrast with the background color. Defaults to True.
        round (int, optional):
            The number of decimal places to round the data to. Defaults to 6.

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

    df = month_hour_binned_series(
        series=utci_data,
        month_bins=month_bins,
        hour_bins=hour_bins,
        month_labels=month_labels,
        hour_labels=hour_labels,
        agg=agg,
    )

    if color_result:
        warnings.warn(
            'The value returned by this method when "color_result" is applied is not a pd.Dataframe object. To get the dataframe use ".data".'
        )

        def _highlight(val):
            bg_color = utci_categories.get_color(val)
            if kwargs.pop("contrasting_text", True):
                fg_color = contrasting_color(bg_color)
            else:
                fg_color = "black"
            return f"color:{fg_color};background-color:{bg_color}"

        return df.style.format(precision=kwargs.pop("round", 6)).applymap(_highlight)

    return df


def met_rate_adjustment(
    utci_collection: HourlyContinuousCollection, met: float
) -> HourlyContinuousCollection:
    """Adjust a UTCI data collection using a target met-rate.

    This method uses the relationshp between UTCI and MET rate described in LINDNER-CENDROWSKA, Katarzyna and BRÖDE, Peter, 2021. The evaluation of biothermal conditions for various forms of climatic therapy based on UTCI adjusted for activity [online]. 2021. IGiPZ PAN. [Accessed 27 October 2022]. Available from: http://rcin.org.pl/igipz/Content/194924/PDF/WA51_229409_r2021-t94-no2_G-Polonica-Linder.pdf.

    +---------------------------------------------------------------+----------------------+
    | Activity                                                      | Metabolic rate (MET) |
    +---------------------------------------------------------------+----------------------+
    | Neutral (resting in sitting or standing position)             | 1.1                  |
    +---------------------------------------------------------------+----------------------+
    | Slow walk (on even path, without load, at 3-4 km·h-1)         | 2.3                  |
    | (default for UTCI calculation)                                |                      |
    +---------------------------------------------------------------+----------------------+
    | Fast walk (on even path, without load, at ~5 km·h-1)          | 3.4                  |
    +---------------------------------------------------------------+----------------------+
    | Marching (on even path, without load, at ~5.5 km·h-1)         | 4.0                  |
    +---------------------------------------------------------------+----------------------+
    | Bicycling (for pleasure, on flat terrain, at < 16 km·h-1)     | 4.0                  |
    +---------------------------------------------------------------+----------------------+
    | Nordic walking (for exercise, on flat terrain, at 5-6 km·h-1) | 4.8                  |
    +---------------------------------------------------------------+----------------------+

    Args:
        utci_collection (HourlyContinuousCollection):
            A UTCI data collection.
        met (float, optional):
            The metabolic rate to apply to this data collection.

    Returns:
        HourlyContinuousCollection:
            An adjusted UTCI data collection.
    """

    if met < 1.1:
        raise ValueError(
            "met_rate must be >= 1.1 (representative of a human body at rest)."
        )
    if met > 4.8:
        raise ValueError(
            "met_rate must be <= 4.8 (representative of a exercise at 5-6km·h-1)."
        )

    # data below extracted from https://doi.org/10.7163/GPol.0199 in the format {MET: [UTCI, ΔUTCI]}
    data = {
        4.8: [
            [-50, 71.81581439393939],
            [-46.47239263803681, 71.5127840909091],
            [-42.94478527607362, 71.36126893939394],
            [-38.95705521472392, 70.14914772727273],
            [-34.96932515337423, 67.421875],
            [-30.521472392638035, 64.69460227272728],
            [-26.993865030674847, 61.66429924242425],
            [-23.006134969325153, 58.33096590909091],
            [-17.944785276073617, 52.87642045454545],
            [-12.576687116564415, 48.63399621212122],
            [-6.441717791411044, 43.63399621212122],
            [-1.380368098159508, 40.300662878787875],
            [4.29447852760736, 35.755208333333336],
            [9.815950920245399, 30.906723484848484],
            [15.490797546012274, 26.512784090909093],
            [23.00613496932516, 20.755208333333336],
            [28.06748466257669, 16.967329545454547],
            [30.98159509202455, 15.452178030303031],
            [35.889570552147234, 12.573390151515156],
            [42.63803680981596, 9.846117424242422],
            [50, 6.967329545454547],
        ],
        4.0: [
            [-50, 46.05823863636364],
            [-47.69938650306749, 47.421875],
            [-45.0920245398773, 48.63399621212122],
            [-43.86503067484662, 50.60369318181819],
            [-40.1840490797546, 51.96732954545455],
            [-36.04294478527608, 51.36126893939394],
            [-33.74233128834356, 50.906723484848484],
            [-30.67484662576687, 49.543087121212125],
            [-27.300613496932513, 47.87642045454545],
            [-24.846625766871163, 46.36126893939394],
            [-23.159509202453986, 44.99763257575758],
            [-20.39877300613497, 42.87642045454545],
            [-16.871165644171782, 39.69460227272728],
            [-14.11042944785276, 37.573390151515156],
            [-11.04294478527607, 35.60369318181819],
            [-7.515337423312886, 34.24005681818182],
            [-3.374233128834355, 32.27035984848485],
            [1.0736196319018418, 29.84611742424243],
            [4.141104294478531, 28.02793560606061],
            [7.055214723926383, 25.45217803030303],
            [9.969325153374236, 23.482481060606062],
            [13.036809815950924, 21.664299242424242],
            [16.41104294478528, 19.543087121212125],
            [19.631901840490798, 17.27035984848485],
            [22.852760736196316, 14.846117424242422],
            [26.22699386503068, 12.724905303030305],
            [29.601226993865026, 11.664299242424242],
            [33.28220858895706, 9.846117424242422],
            [36.65644171779141, 8.785511363636367],
            [39.87730061349693, 8.330965909090907],
            [42.63803680981596, 7.421875],
            [46.012269938650306, 6.664299242424242],
            [50, 5.452178030303031],
        ],
        3.4: [
            [-50, 27.724905303030305],
            [-47.239263803680984, 28.785511363636367],
            [-44.47852760736196, 29.543087121212125],
            [-41.717791411042946, 30.45217803030303],
            [-38.34355828220859, 31.05823863636364],
            [-35.2760736196319, 31.05823863636364],
            [-33.43558282208589, 31.361268939393938],
            [-30.98159509202454, 31.815814393939398],
            [-28.220858895705522, 31.967329545454547],
            [-26.380368098159508, 31.05823863636364],
            [-24.846625766871163, 30.300662878787882],
            [-23.006134969325153, 29.694602272727273],
            [-21.165644171779142, 28.482481060606062],
            [-18.558282208588956, 26.967329545454547],
            [-16.411042944785272, 25.755208333333336],
            [-14.11042944785276, 24.39157196969697],
            [-11.65644171779141, 23.02793560606061],
            [-8.742331288343557, 23.179450757575758],
            [-5.828220858895705, 23.330965909090914],
            [-2.4539877300613497, 22.573390151515156],
            [0.6134969325153392, 21.361268939393938],
            [3.374233128834355, 19.846117424242422],
            [6.74846625766871, 17.876420454545453],
            [9.662576687116562, 16.512784090909093],
            [12.883435582822088, 14.846117424242422],
            [16.104294478527606, 14.088541666666664],
            [19.631901840490798, 11.815814393939398],
            [22.085889570552155, 10.300662878787882],
            [24.84662576687117, 8.785511363636367],
            [28.52760736196319, 8.027935606060609],
            [31.44171779141105, 7.1188446969696955],
            [34.20245398773007, 6.058238636363637],
            [37.269938650306756, 5.755208333333336],
            [40.03067484662577, 5.452178030303031],
            [42.94478527607362, 4.846117424242426],
            [46.012269938650306, 4.088541666666668],
            [50, 3.9370265151515156],
        ],
        2.3: [
            [-50, -0.15151515151515227],
            [50, 0],
        ],
        1.1: [
            [-50, -27.87878787878788],
            [-48.15950920245399, -26.96969696969697],
            [-45.858895705521476, -26.21212121212121],
            [-44.6319018404908, -24.09090909090909],
            [-42.484662576687114, -21.96969696969697],
            [-39.57055214723926, -19.848484848484848],
            [-36.04294478527608, -19.242424242424242],
            [-32.20858895705521, -20],
            [-29.447852760736197, -21.363636363636363],
            [-25.460122699386503, -23.03030303030303],
            [-22.54601226993865, -24.848484848484848],
            [-19.32515337423313, -26.666666666666668],
            [-16.257668711656443, -28.484848484848484],
            [-13.34355828220859, -30.151515151515152],
            [-9.509202453987726, -30.90909090909091],
            [-5.981595092024541, -31.060606060606062],
            [-3.067484662576689, -29.393939393939394],
            [-1.380368098159508, -27.272727272727273],
            [0, -24.848484848484848],
            [1.0736196319018418, -21.666666666666668],
            [2.760736196319016, -18.78787878787879],
            [4.141104294478531, -15.909090909090908],
            [6.74846625766871, -13.484848484848484],
            [9.969325153374236, -11.818181818181818],
            [13.190184049079754, -11.363636363636363],
            [16.257668711656436, -9.242424242424242],
            [19.785276073619627, -9.696969696969697],
            [23.00613496932516, -10.303030303030303],
            [26.380368098159508, -9.242424242424242],
            [29.447852760736197, -7.272727272727273],
            [32.515337423312886, -6.363636363636363],
            [35.582822085889575, -5.303030303030301],
            [38.650306748466264, -5.151515151515152],
            [41.25766871165645, -4.3939393939393945],
            [44.018404907975466, -3.787878787878789],
            [46.31901840490798, -3.333333333333332],
            [50, -3.6363636363636367],
        ],
    }

    matrix = []
    for k, v in data.items():
        x, y = np.array(v).T
        f = interp1d(x, y)
        new_x = np.linspace(-50, 50, 1000)
        new_y = f(new_x)
        matrix.append(pd.Series(index=new_x, data=new_y, name=k))
    met_rate, utci_val, utci_delta = (
        pd.concat(matrix, axis=1).unstack().reset_index().values.T
    )

    # create 2d interpolator between [MET, UTCI] and [ΔUTCI]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster = interp2d(met_rate, utci_val, utci_delta)

    # Calculate ΔUTCI
    original_utci = collection_to_series(utci_collection)
    utci_delta = [forecaster(met, i)[0] for i in original_utci.values]

    return collection_from_series(original_utci + utci_delta)
