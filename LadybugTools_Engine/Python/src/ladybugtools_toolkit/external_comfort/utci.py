import calendar
import warnings
from calendar import month_name
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import OutdoorSolarCal
from ladybug_comfort.collection.utci import UTCI
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, to_rgba
from numpy.typing import NDArray
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm

from ..ladybug_extension.analysis_period import describe as describe_analysis_period
from ..ladybug_extension.analysis_period import to_boolean
from ..ladybug_extension.datacollection import from_series, to_series
from ..ladybug_extension.epw import (
    seasonality_from_day_length,
    seasonality_from_month,
    seasonality_from_temperature,
)
from ..plot.colormaps import UTCI_LABELS
from .moisture import evaporative_cooling_effect_collection


class UniversalThermalClimateIndex(Enum):
    EXTREME_COLD_STRESS = "Extreme cold stress"
    VERY_STRONG_COLD_STRESS = "Very strong cold stress"
    STRONG_COLD_STRESS = "Strong cold stress"
    MODERATE_COLD_STRESS = "Moderate cold stress"
    SLIGHT_COLD_STRESS = "Slight cold stress"
    NO_THERMAL_STRESS = "No thermal stress"
    MODERATE_HEAT_STRESS = "Moderate heat stress"
    STRONG_HEAT_STRESS = "Strong heat stress"
    VERY_STRONG_HEAT_STRESS = "Very strong heat stress"
    EXTREME_HEAT_STRESS = "Extreme heat stress"

    @property
    def low_limit(self) -> float:
        """The low-threshold for the UTCI category."""
        d = {
            self.EXTREME_COLD_STRESS.value: -np.inf,
            self.VERY_STRONG_COLD_STRESS.value: -40,
            self.STRONG_COLD_STRESS.value: -27,
            self.MODERATE_COLD_STRESS.value: -13,
            self.SLIGHT_COLD_STRESS.value: 0,
            self.NO_THERMAL_STRESS.value: 9,
            self.MODERATE_HEAT_STRESS.value: 26,
            self.STRONG_HEAT_STRESS.value: 32,
            self.VERY_STRONG_HEAT_STRESS.value: 38,
            self.EXTREME_HEAT_STRESS.value: 46,
        }
        return d[self.value]

    @property
    def high_limit(self) -> float:
        """The high-threshold for the UTCI category."""
        d = {
            self.EXTREME_COLD_STRESS.value: -40,
            self.VERY_STRONG_COLD_STRESS.value: -27,
            self.STRONG_COLD_STRESS.value: -13,
            self.MODERATE_COLD_STRESS.value: 0,
            self.SLIGHT_COLD_STRESS.value: 9,
            self.NO_THERMAL_STRESS.value: 26,
            self.MODERATE_HEAT_STRESS.value: 32,
            self.STRONG_HEAT_STRESS.value: 38,
            self.VERY_STRONG_HEAT_STRESS.value: 46,
            self.EXTREME_HEAT_STRESS.value: np.inf,
        }
        return d[self.value]

    @property
    def range(self) -> Tuple[float]:
        """The thresholds for the UTCI category."""

        return (self.low_limit, self.high_limit)

    @property
    def color(self) -> str:
        """The color associated with the UTCI category."""
        d = {
            self.EXTREME_COLD_STRESS.value: "#0D104B",
            self.VERY_STRONG_COLD_STRESS.value: "#262972",
            self.STRONG_COLD_STRESS.value: "#3452A4",
            self.MODERATE_COLD_STRESS.value: "#3C65AF",
            self.SLIGHT_COLD_STRESS.value: "#37BCED",
            self.NO_THERMAL_STRESS.value: "#2EB349",
            self.MODERATE_HEAT_STRESS.value: "#F38322",
            self.STRONG_HEAT_STRESS.value: "#C31F25",
            self.VERY_STRONG_HEAT_STRESS.value: "#7F1416",
            self.EXTREME_HEAT_STRESS.value: "#580002",
        }
        return d[self.value]

    @property
    def description(self) -> str:
        """Human readable description of this comfort category."""
        if np.isinf(self.low_limit):
            return f"{self.value} (<{self.high_limit}°C UTCI)"

        if np.isinf(self.high_limit):
            return f"{self.value} (>{self.low_limit}°C UTCI)"

        return f"{self.value} ({self.low_limit}°C < x < {self.high_limit}°C UTCI)"


def cmap() -> Colormap:
    """Return the colormap associated with this comfort metric."""
    utci_cmap = ListedColormap(
        colors=[i.color for i in UniversalThermalClimateIndex],
        name=f"{UniversalThermalClimateIndex.__name__}",
    )
    utci_cmap.set_under(UniversalThermalClimateIndex.EXTREME_COLD_STRESS.color)
    utci_cmap.set_over(UniversalThermalClimateIndex.EXTREME_HEAT_STRESS.color)

    return utci_cmap


def boundarynorm() -> BoundaryNorm:
    """Return the boundary-norm associate with this comfort metric."""
    return BoundaryNorm(
        np.unique([i.range for i in UniversalThermalClimateIndex])[1:-1], cmap().N
    )


def utci(
    air_temperature: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    relative_humidity: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    mean_radiant_temperature: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
    wind_speed: Union[
        HourlyContinuousCollection, pd.DataFrame, pd.Series, NDArray[np.float64]
    ],
) -> Union[HourlyContinuousCollection, pd.DataFrame, NDArray[np.float64]]:
    """Return the UTCI for the given inputs.

    Returns:
        HourlyContinuousCollection: The calculated UTCI based on the shelter configuration for the given typology.
    """
    _inputs = [
        air_temperature,
        relative_humidity,
        mean_radiant_temperature,
        wind_speed,
    ]

    if all((isinstance(i, HourlyContinuousCollection) for i in _inputs)):
        return UTCI(
            air_temperature=air_temperature,
            rel_humidity=relative_humidity,
            rad_temperature=mean_radiant_temperature,
            wind_speed=wind_speed,
        ).universal_thermal_climate_index

    if all((isinstance(i, (float, int)) for i in _inputs)):
        return utci_vectorised(
            ta=air_temperature,
            rh=relative_humidity,
            tr=mean_radiant_temperature,
            vel=np.clip([wind_speed], 0, 17)[0],
        )

    if all((isinstance(i, pd.DataFrame) for i in _inputs)):
        return pd.DataFrame(
            utci_vectorised(
                ta=air_temperature.values,
                rh=relative_humidity.values,
                tr=mean_radiant_temperature.values,
                vel=wind_speed.clip(lower=0, upper=17).values,
            ),
            columns=_inputs[0].columns
            if len(_inputs[0].columns) > 1
            else ["Universal Thermal Climate Index (C)"],
            index=_inputs[0].index,
        )

    if all((isinstance(i, pd.Series) for i in _inputs)):
        return pd.Series(
            utci_vectorised(
                ta=air_temperature,
                rh=relative_humidity,
                tr=mean_radiant_temperature,
                vel=wind_speed.clip(lower=0, upper=17),
            ),
            name="Universal Thermal Climate Index (C)",
            index=_inputs[0].index,
        )

    if all((isinstance(i, (List, Tuple)) for i in _inputs)):
        return utci_vectorised(
            ta=np.array(air_temperature),
            rh=np.array(relative_humidity),
            tr=np.array(mean_radiant_temperature),
            vel=np.clip(np.array(wind_speed), 0, 17),
        )

    raise ValueError(
        "No possible means of calculating UTCI from that combination of inputs was found."
    )


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

    # total_number_of_hours = len(col_baseline)
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
    comfortable_hours_worse = comfortable_hours_difference > 0
    hot_hours_difference = hot_hours_baseline - hot_hours_comparable
    hot_hours_worse = hot_hours_difference < 0
    cold_hours_difference = cold_hours_baseline - cold_hours_comparable
    cold_hours_worse = cold_hours_difference < 0

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


def describe_as_dataframe(
    universal_thermal_climate_index: HourlyContinuousCollection,
    analysis_periods: Tuple[AnalysisPeriod] = (AnalysisPeriod()),
    comfort_limits: Tuple[float] = (9, 26),
) -> pd.DataFrame:
    """Create a text summary of the given UTCI data collection.

    Args:
        universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing UTCI values.
        analysis_periods (List[AnalysisPeriod], optional):
            A list of analysis periods over which to summaries the collection. Defaults to AnalysisPeriod().
        comfort_limits (Tuple[float], optional):
            Bespoke comfort limits. Defaults to (9, 26).

    Returns:
        pd.DataFrame:
            A table containing % comfort for given analysis periods.
    """

    limit_low = min(comfort_limits)
    limit_high = max(comfort_limits)

    index = [
        "Hours",
        f"No Thermal Stress [{limit_low} <= x <= {limit_high}] (Hours)",
        f"Heat Stress [x > {limit_high}] (Hours)",
        f"Cold Stress [x < {limit_low}] (Hours)",
        f"No Thermal Stress [{limit_low} <= x <= {limit_high}]",
        f"Heat Stress [x > {limit_high}]",
        f"Cold Stress [x < {limit_low}]",
    ]

    dfs = []
    for analysis_period in analysis_periods:
        col = universal_thermal_climate_index.filter_by_analysis_period(analysis_period)
        series = to_series(col)

        total_number_of_hours = len(series)
        comfortable_hours = ((series >= limit_low) & (series <= limit_high)).sum()
        hot_hours = (series > limit_high).sum()
        cold_hours = (series < limit_low).sum()
        comfortable_hours_percentage = comfortable_hours / total_number_of_hours
        hot_hours_percentage = hot_hours / total_number_of_hours
        cold_hours_percentage = cold_hours / total_number_of_hours

        dfs.append(
            pd.Series(
                name=analysis_period,
                data=[
                    total_number_of_hours,
                    comfortable_hours,
                    hot_hours,
                    cold_hours,
                    comfortable_hours_percentage,
                    hot_hours_percentage,
                    cold_hours_percentage,
                ],
                index=index,
            )
        )
    df = pd.concat(dfs, axis=1)

    return df.T


def describe(
    universal_thermal_climate_index: HourlyContinuousCollection,
    analysis_period: AnalysisPeriod = None,
    comfort_limits: Tuple[float] = (9, 26),
    sep: str = "\n",
) -> str:
    """Create a text summary of the given UTCI data collection.

    Args:
        universal_thermal_climate_index (HourlyContinuousCollection):
            A collection containing UTCI values.
        analysis_period (AnalysisPeriod, optional):
            An analysis period over which to summaries the collection. Defaults to None.
        comfort_limits (Tuple[float], optional):
            Bespoke comfort limits. Defaults to (9, 26).
        sep: (str, optional):
            A separator for each summary string. Default is "\n".

    Returns:
        str:
            A text summary of the given UTCI data collection.
    """
    if analysis_period is None:
        analysis_period = AnalysisPeriod()

    ap_description = describe_analysis_period(analysis_period, include_timestep=False)

    col = universal_thermal_climate_index.filter_by_analysis_period(analysis_period)
    series = to_series(col)

    limit_low = min(comfort_limits)
    limit_high = max(comfort_limits)

    total_number_of_hours = len(series)
    comfortable_hours = ((series >= limit_low) & (series <= limit_high)).sum()
    hot_hours = (series > limit_high).sum()
    cold_hours = (series < limit_low).sum()

    statements = [
        f"In this summary, thermal comfort or periods experiencing no thermal stress, are UTCI values of between {limit_low}°C and {limit_high}°C.",
        f'For {ap_description}, "No thermal stress" is expected for {comfortable_hours} out of a possible {total_number_of_hours} hours ({comfortable_hours/total_number_of_hours:0.1%}).',
        f'"Cold stress" is expected for {cold_hours} hours ({cold_hours/total_number_of_hours:0.1%}).',
        f'"Heat stress" is expected for {hot_hours} hours ({hot_hours/total_number_of_hours:0.1%}).',
    ]
    return sep.join(statements)


def describe_monthly(
    utci_collection: HourlyContinuousCollection,
    comfort_limits: Tuple[float] = (9, 26),
    density: bool = True,
    annual_mask: List[bool] = np.ones(8760).astype(bool),
) -> pd.DataFrame:
    """Create a monthly table containing cold/comfortable/hot comfort categories for each month."""

    # convert
    s = to_series(utci_collection)

    # filter for hours
    s = s[annual_mask]

    # get counts
    month_counts = s.groupby(s.index.month).count()

    # calculate metrics
    cold = (s < min(comfort_limits)).groupby(s.index.month).sum()
    hot = (s > max(comfort_limits)).groupby(s.index.month).sum()
    comfortable = month_counts - cold - hot

    # convert to percentage if density == True
    if density:
        cold = cold / month_counts
        hot = hot / month_counts
        comfortable = comfortable / month_counts

    df = pd.concat(
        [cold, comfortable, hot],
        axis=1,
        keys=[
            f"Too cold (<{min(comfort_limits)})",
            "Comfortable",
            f"Too Hot (>{max(comfort_limits)})",
        ],
    )
    df.index = [month_name[i] for i in cold.index]

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
    original_utci = to_series(utci_collection)
    utci_delta = [forecaster(met, i)[0] for i in original_utci.values]

    return from_series(original_utci + utci_delta)


def utci_parallel(
    ta: np.ndarray, tr: np.ndarray, vel: np.ndarray, rh: np.ndarray
) -> np.ndarray:
    """Calculate UTCI a bit faster!

    Args:
        ta (np.ndarray):
            Air temperature [C]
        tr (np.ndarray):
            Mean radiant temperature [C]
        vel (np.ndarray):
            Wind speed 10m above ground level [m/s]
        rh (np.ndarray):
            Relative humidity [%]

    Returns:
        np.ndarray:
            UTCI values.
    """

    if ta.shape != tr.shape != vel.shape != rh.shape:
        raise ValueError("Input arrays must be of the same shape.")
    if any(
        [
            len(ta.shape) != 2,
            len(tr.shape) != 2,
            len(vel.shape) != 2,
            len(rh.shape) != 2,
        ]
    ):
        raise ValueError("Input arrays must be of shape (n, m).")

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(utci_vectorised, ta, tr, vel, rh),
                total=len(ta),
                desc="Calculating UTCI: ",
            )
        )

    return np.concatenate(results).reshape(ta.shape)


def utci_vectorised(
    ta: np.ndarray, tr: np.ndarray, vel: np.ndarray, rh: np.ndarray
) -> np.ndarray:
    """This method is a vectorised version of the universal_thermal_climate_index method defined in ladybug-tools
    https://github.com/ladybug-tools/ladybug-comfort/blob/master/ladybug_comfort/utci.py

    Args:
        ta (np.ndarray): Air temperature [C]
        tr (np.ndarray): Mean radiant temperature [C]
        vel (np.ndarray): Wind speed 10 m above ground level [m/s]
        rh (np.ndarray): Relative humidity [%]

    Returns:
        np.ndarray: The Universal Thermal Climate Index (UTCI) for the input conditions as approximated by a 4-D polynomial
    """

    g = (
        -2836.5744,
        -6028.076559,
        19.54263612,
        -0.02737830188,
        0.000016261698,
        7.0229056e-10,
        -1.8680009e-13,
    )
    tk = ta + 273.15  # air temp in K
    es = 2.7150305 * np.log(tk)
    for i, x in enumerate(g):
        es = es + (x * (tk ** (i - 2)))
    es = np.exp(es) * 0.01
    eh_pa = es * (rh / 100.0)  # partial vapor pressure
    pa_pr = eh_pa / 10.0  # convert vapour pressure to kPa
    d_tr = tr - ta  # difference between radiant and air temperature

    utci_approx = (
        ta
        + 0.607562052
        + -0.0227712343 * ta
        + 8.06470249e-4 * ta * ta
        + -1.54271372e-4 * ta * ta * ta
        + -3.24651735e-6 * ta * ta * ta * ta
        + 7.32602852e-8 * ta * ta * ta * ta * ta
        + 1.35959073e-9 * ta * ta * ta * ta * ta * ta
        + -2.25836520 * vel
        + 0.0880326035 * ta * vel
        + 0.00216844454 * ta * ta * vel
        + -1.53347087e-5 * ta * ta * ta * vel
        + -5.72983704e-7 * ta * ta * ta * ta * vel
        + -2.55090145e-9 * ta * ta * ta * ta * ta * vel
        + -0.751269505 * vel * vel
        + -0.00408350271 * ta * vel * vel
        + -5.21670675e-5 * ta * ta * vel * vel
        + 1.94544667e-6 * ta * ta * ta * vel * vel
        + 1.14099531e-8 * ta * ta * ta * ta * vel * vel
        + 0.158137256 * vel * vel * vel
        + -6.57263143e-5 * ta * vel * vel * vel
        + 2.22697524e-7 * ta * ta * vel * vel * vel
        + -4.16117031e-8 * ta * ta * ta * vel * vel * vel
        + -0.0127762753 * vel * vel * vel * vel
        + 9.66891875e-6 * ta * vel * vel * vel * vel
        + 2.52785852e-9 * ta * ta * vel * vel * vel * vel
        + 4.56306672e-4 * vel * vel * vel * vel * vel
        + -1.74202546e-7 * ta * vel * vel * vel * vel * vel
        + -5.91491269e-6 * vel * vel * vel * vel * vel * vel
        + 0.398374029 * d_tr
        + 1.83945314e-4 * ta * d_tr
        + -1.73754510e-4 * ta * ta * d_tr
        + -7.60781159e-7 * ta * ta * ta * d_tr
        + 3.77830287e-8 * ta * ta * ta * ta * d_tr
        + 5.43079673e-10 * ta * ta * ta * ta * ta * d_tr
        + -0.0200518269 * vel * d_tr
        + 8.92859837e-4 * ta * vel * d_tr
        + 3.45433048e-6 * ta * ta * vel * d_tr
        + -3.77925774e-7 * ta * ta * ta * vel * d_tr
        + -1.69699377e-9 * ta * ta * ta * ta * vel * d_tr
        + 1.69992415e-4 * vel * vel * d_tr
        + -4.99204314e-5 * ta * vel * vel * d_tr
        + 2.47417178e-7 * ta * ta * vel * vel * d_tr
        + 1.07596466e-8 * ta * ta * ta * vel * vel * d_tr
        + 8.49242932e-5 * vel * vel * vel * d_tr
        + 1.35191328e-6 * ta * vel * vel * vel * d_tr
        + -6.21531254e-9 * ta * ta * vel * vel * vel * d_tr
        + -4.99410301e-6 * vel * vel * vel * vel * d_tr
        + -1.89489258e-8 * ta * vel * vel * vel * vel * d_tr
        + 8.15300114e-8 * vel * vel * vel * vel * vel * d_tr
        + 7.55043090e-4 * d_tr * d_tr
        + -5.65095215e-5 * ta * d_tr * d_tr
        + -4.52166564e-7 * ta * ta * d_tr * d_tr
        + 2.46688878e-8 * ta * ta * ta * d_tr * d_tr
        + 2.42674348e-10 * ta * ta * ta * ta * d_tr * d_tr
        + 1.54547250e-4 * vel * d_tr * d_tr
        + 5.24110970e-6 * ta * vel * d_tr * d_tr
        + -8.75874982e-8 * ta * ta * vel * d_tr * d_tr
        + -1.50743064e-9 * ta * ta * ta * vel * d_tr * d_tr
        + -1.56236307e-5 * vel * vel * d_tr * d_tr
        + -1.33895614e-7 * ta * vel * vel * d_tr * d_tr
        + 2.49709824e-9 * ta * ta * vel * vel * d_tr * d_tr
        + 6.51711721e-7 * vel * vel * vel * d_tr * d_tr
        + 1.94960053e-9 * ta * vel * vel * vel * d_tr * d_tr
        + -1.00361113e-8 * vel * vel * vel * vel * d_tr * d_tr
        + -1.21206673e-5 * d_tr * d_tr * d_tr
        + -2.18203660e-7 * ta * d_tr * d_tr * d_tr
        + 7.51269482e-9 * ta * ta * d_tr * d_tr * d_tr
        + 9.79063848e-11 * ta * ta * ta * d_tr * d_tr * d_tr
        + 1.25006734e-6 * vel * d_tr * d_tr * d_tr
        + -1.81584736e-9 * ta * vel * d_tr * d_tr * d_tr
        + -3.52197671e-10 * ta * ta * vel * d_tr * d_tr * d_tr
        + -3.36514630e-8 * vel * vel * d_tr * d_tr * d_tr
        + 1.35908359e-10 * ta * vel * vel * d_tr * d_tr * d_tr
        + 4.17032620e-10 * vel * vel * vel * d_tr * d_tr * d_tr
        + -1.30369025e-9 * d_tr * d_tr * d_tr * d_tr
        + 4.13908461e-10 * ta * d_tr * d_tr * d_tr * d_tr
        + 9.22652254e-12 * ta * ta * d_tr * d_tr * d_tr * d_tr
        + -5.08220384e-9 * vel * d_tr * d_tr * d_tr * d_tr
        + -2.24730961e-11 * ta * vel * d_tr * d_tr * d_tr * d_tr
        + 1.17139133e-10 * vel * vel * d_tr * d_tr * d_tr * d_tr
        + 6.62154879e-10 * d_tr * d_tr * d_tr * d_tr * d_tr
        + 4.03863260e-13 * ta * d_tr * d_tr * d_tr * d_tr * d_tr
        + 1.95087203e-12 * vel * d_tr * d_tr * d_tr * d_tr * d_tr
        + -4.73602469e-12 * d_tr * d_tr * d_tr * d_tr * d_tr * d_tr
        + 5.12733497 * pa_pr
        + -0.312788561 * ta * pa_pr
        + -0.0196701861 * ta * ta * pa_pr
        + 9.99690870e-4 * ta * ta * ta * pa_pr
        + 9.51738512e-6 * ta * ta * ta * ta * pa_pr
        + -4.66426341e-7 * ta * ta * ta * ta * ta * pa_pr
        + 0.548050612 * vel * pa_pr
        + -0.00330552823 * ta * vel * pa_pr
        + -0.00164119440 * ta * ta * vel * pa_pr
        + -5.16670694e-6 * ta * ta * ta * vel * pa_pr
        + 9.52692432e-7 * ta * ta * ta * ta * vel * pa_pr
        + -0.0429223622 * vel * vel * pa_pr
        + 0.00500845667 * ta * vel * vel * pa_pr
        + 1.00601257e-6 * ta * ta * vel * vel * pa_pr
        + -1.81748644e-6 * ta * ta * ta * vel * vel * pa_pr
        + -1.25813502e-3 * vel * vel * vel * pa_pr
        + -1.79330391e-4 * ta * vel * vel * vel * pa_pr
        + 2.34994441e-6 * ta * ta * vel * vel * vel * pa_pr
        + 1.29735808e-4 * vel * vel * vel * vel * pa_pr
        + 1.29064870e-6 * ta * vel * vel * vel * vel * pa_pr
        + -2.28558686e-6 * vel * vel * vel * vel * vel * pa_pr
        + -0.0369476348 * d_tr * pa_pr
        + 0.00162325322 * ta * d_tr * pa_pr
        + -3.14279680e-5 * ta * ta * d_tr * pa_pr
        + 2.59835559e-6 * ta * ta * ta * d_tr * pa_pr
        + -4.77136523e-8 * ta * ta * ta * ta * d_tr * pa_pr
        + 8.64203390e-3 * vel * d_tr * pa_pr
        + -6.87405181e-4 * ta * vel * d_tr * pa_pr
        + -9.13863872e-6 * ta * ta * vel * d_tr * pa_pr
        + 5.15916806e-7 * ta * ta * ta * vel * d_tr * pa_pr
        + -3.59217476e-5 * vel * vel * d_tr * pa_pr
        + 3.28696511e-5 * ta * vel * vel * d_tr * pa_pr
        + -7.10542454e-7 * ta * ta * vel * vel * d_tr * pa_pr
        + -1.24382300e-5 * vel * vel * vel * d_tr * pa_pr
        + -7.38584400e-9 * ta * vel * vel * vel * d_tr * pa_pr
        + 2.20609296e-7 * vel * vel * vel * vel * d_tr * pa_pr
        + -7.32469180e-4 * d_tr * d_tr * pa_pr
        + -1.87381964e-5 * ta * d_tr * d_tr * pa_pr
        + 4.80925239e-6 * ta * ta * d_tr * d_tr * pa_pr
        + -8.75492040e-8 * ta * ta * ta * d_tr * d_tr * pa_pr
        + 2.77862930e-5 * vel * d_tr * d_tr * pa_pr
        + -5.06004592e-6 * ta * vel * d_tr * d_tr * pa_pr
        + 1.14325367e-7 * ta * ta * vel * d_tr * d_tr * pa_pr
        + 2.53016723e-6 * vel * vel * d_tr * d_tr * pa_pr
        + -1.72857035e-8 * ta * vel * vel * d_tr * d_tr * pa_pr
        + -3.95079398e-8 * vel * vel * vel * d_tr * d_tr * pa_pr
        + -3.59413173e-7 * d_tr * d_tr * d_tr * pa_pr
        + 7.04388046e-7 * ta * d_tr * d_tr * d_tr * pa_pr
        + -1.89309167e-8 * ta * ta * d_tr * d_tr * d_tr * pa_pr
        + -4.79768731e-7 * vel * d_tr * d_tr * d_tr * pa_pr
        + 7.96079978e-9 * ta * vel * d_tr * d_tr * d_tr * pa_pr
        + 1.62897058e-9 * vel * vel * d_tr * d_tr * d_tr * pa_pr
        + 3.94367674e-8 * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -1.18566247e-9 * ta * d_tr * d_tr * d_tr * d_tr * pa_pr
        + 3.34678041e-10 * vel * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -1.15606447e-10 * d_tr * d_tr * d_tr * d_tr * d_tr * pa_pr
        + -2.80626406 * pa_pr * pa_pr
        + 0.548712484 * ta * pa_pr * pa_pr
        + -0.00399428410 * ta * ta * pa_pr * pa_pr
        + -9.54009191e-4 * ta * ta * ta * pa_pr * pa_pr
        + 1.93090978e-5 * ta * ta * ta * ta * pa_pr * pa_pr
        + -0.308806365 * vel * pa_pr * pa_pr
        + 0.0116952364 * ta * vel * pa_pr * pa_pr
        + 4.95271903e-4 * ta * ta * vel * pa_pr * pa_pr
        + -1.90710882e-5 * ta * ta * ta * vel * pa_pr * pa_pr
        + 0.00210787756 * vel * vel * pa_pr * pa_pr
        + -6.98445738e-4 * ta * vel * vel * pa_pr * pa_pr
        + 2.30109073e-5 * ta * ta * vel * vel * pa_pr * pa_pr
        + 4.17856590e-4 * vel * vel * vel * pa_pr * pa_pr
        + -1.27043871e-5 * ta * vel * vel * vel * pa_pr * pa_pr
        + -3.04620472e-6 * vel * vel * vel * vel * pa_pr * pa_pr
        + 0.0514507424 * d_tr * pa_pr * pa_pr
        + -0.00432510997 * ta * d_tr * pa_pr * pa_pr
        + 8.99281156e-5 * ta * ta * d_tr * pa_pr * pa_pr
        + -7.14663943e-7 * ta * ta * ta * d_tr * pa_pr * pa_pr
        + -2.66016305e-4 * vel * d_tr * pa_pr * pa_pr
        + 2.63789586e-4 * ta * vel * d_tr * pa_pr * pa_pr
        + -7.01199003e-6 * ta * ta * vel * d_tr * pa_pr * pa_pr
        + -1.06823306e-4 * vel * vel * d_tr * pa_pr * pa_pr
        + 3.61341136e-6 * ta * vel * vel * d_tr * pa_pr * pa_pr
        + 2.29748967e-7 * vel * vel * vel * d_tr * pa_pr * pa_pr
        + 3.04788893e-4 * d_tr * d_tr * pa_pr * pa_pr
        + -6.42070836e-5 * ta * d_tr * d_tr * pa_pr * pa_pr
        + 1.16257971e-6 * ta * ta * d_tr * d_tr * pa_pr * pa_pr
        + 7.68023384e-6 * vel * d_tr * d_tr * pa_pr * pa_pr
        + -5.47446896e-7 * ta * vel * d_tr * d_tr * pa_pr * pa_pr
        + -3.59937910e-8 * vel * vel * d_tr * d_tr * pa_pr * pa_pr
        + -4.36497725e-6 * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 1.68737969e-7 * ta * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 2.67489271e-8 * vel * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + 3.23926897e-9 * d_tr * d_tr * d_tr * d_tr * pa_pr * pa_pr
        + -0.0353874123 * pa_pr * pa_pr * pa_pr
        + -0.221201190 * ta * pa_pr * pa_pr * pa_pr
        + 0.0155126038 * ta * ta * pa_pr * pa_pr * pa_pr
        + -2.63917279e-4 * ta * ta * ta * pa_pr * pa_pr * pa_pr
        + 0.0453433455 * vel * pa_pr * pa_pr * pa_pr
        + -0.00432943862 * ta * vel * pa_pr * pa_pr * pa_pr
        + 1.45389826e-4 * ta * ta * vel * pa_pr * pa_pr * pa_pr
        + 2.17508610e-4 * vel * vel * pa_pr * pa_pr * pa_pr
        + -6.66724702e-5 * ta * vel * vel * pa_pr * pa_pr * pa_pr
        + 3.33217140e-5 * vel * vel * vel * pa_pr * pa_pr * pa_pr
        + -0.00226921615 * d_tr * pa_pr * pa_pr * pa_pr
        + 3.80261982e-4 * ta * d_tr * pa_pr * pa_pr * pa_pr
        + -5.45314314e-9 * ta * ta * d_tr * pa_pr * pa_pr * pa_pr
        + -7.96355448e-4 * vel * d_tr * pa_pr * pa_pr * pa_pr
        + 2.53458034e-5 * ta * vel * d_tr * pa_pr * pa_pr * pa_pr
        + -6.31223658e-6 * vel * vel * d_tr * pa_pr * pa_pr * pa_pr
        + 3.02122035e-4 * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + -4.77403547e-6 * ta * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + 1.73825715e-6 * vel * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + -4.09087898e-7 * d_tr * d_tr * d_tr * pa_pr * pa_pr * pa_pr
        + 0.614155345 * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.0616755931 * ta * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00133374846 * ta * ta * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00355375387 * vel * pa_pr * pa_pr * pa_pr * pa_pr
        + -5.13027851e-4 * ta * vel * pa_pr * pa_pr * pa_pr * pa_pr
        + 1.02449757e-4 * vel * vel * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.00148526421 * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -4.11469183e-5 * ta * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -6.80434415e-6 * vel * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + -9.77675906e-6 * d_tr * d_tr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.0882773108 * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + -0.00301859306 * ta * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00104452989 * vel * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 2.47090539e-4 * d_tr * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
        + 0.00148348065 * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr * pa_pr
    )

    return utci_approx


def categorise_shade_benefit(
    *,
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
        comfort_limits = tuple(UniversalThermalClimateIndex.NO_THERMAL_STRESS.range)
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
    values: Union[List[float], float], fmt: str = "category", simplified: bool = False
) -> Union[pd.Categorical, str]:
    """Convert a numeric values into their associated UTCI categories."""

    if simplified:
        bins = np.array([-np.inf, 9, 26, np.inf])
    else:
        bins = np.unique([cat.range for cat in UniversalThermalClimateIndex])

    format_options = ["category", "rgba", "hex"]
    if fmt == "hex":
        if simplified:
            labels = ["#3C65AF", "#2EB349", "#C31F25"]
        else:
            labels = [cat.color for cat in UniversalThermalClimateIndex]
    elif fmt == "rgba":
        if simplified:
            labels = [to_rgba(i) for i in ["#3C65AF", "#2EB349", "#C31F25"]]
        else:
            labels = [
                to_rgba(i) for i in [cat.color for cat in UniversalThermalClimateIndex]
            ]
    elif fmt == "category":
        if simplified:
            labels = ["Too cold", "Comfortable", "Too hot"]
        else:
            labels = [cat.value for cat in UniversalThermalClimateIndex]
    else:
        raise ValueError(f"format must be one of {format_options}")

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
    epw: EPW, as_dataframe: bool = False
) -> List[HourlyContinuousCollection]:
    """Calculate the absolute min/max collections of UTCI based on possible shade, wind and moisture conditions.

    Args:
        epw (EPW): The EPW object for which limits will be calculated.
        as_dataframe (bool): Return the output as a dataframe with two columns, instread of two separate collections.

    Returns:
        List[HourlyContinuousCollection]: The lowest UTCI and highest UTCI temperatures for each hour of the year.
    """
    mrt_unshaded = OutdoorSolarCal(
        epw.location,
        epw.direct_normal_radiation,
        epw.diffuse_horizontal_radiation,
        epw.horizontal_infrared_radiation_intensity,
        epw.dry_bulb_temperature,
    ).mean_radiant_temperature
    dbt_evap, rh_evap = evaporative_cooling_effect_collection(
        epw, evaporative_cooling_effectiveness=0.5
    )

    # weatherfile wind, perect shade, evaporatively cooled air
    min_utci = UTCI(
        air_temperature=dbt_evap,
        rad_temperature=epw.dry_bulb_temperature,
        rel_humidity=rh_evap,
        wind_speed=epw.wind_speed,
    ).universal_thermal_climate_index

    # max UTCI - no wind, no shade, no additional moisture in air
    max_utci = UTCI(
        air_temperature=epw.dry_bulb_temperature,
        rad_temperature=mrt_unshaded,
        rel_humidity=epw.relative_humidity,
        wind_speed=epw.wind_speed.get_aligned_collection(0),
    ).universal_thermal_climate_index

    # get the low-high values for each hour, and create the new collections
    abs_min = np.min([min_utci.values, max_utci.values], axis=0)
    abs_max = np.max([min_utci.values, max_utci.values], axis=0)

    if as_dataframe:
        return pd.concat(
            [
                to_series(min_utci.get_aligned_collection(abs_min)),
                to_series(max_utci.get_aligned_collection(abs_max)),
            ],
            axis=1,
            keys=["min", "max"],
        )

    return min_utci.get_aligned_collection(abs_min), max_utci.get_aligned_collection(
        abs_max
    )


def feasible_comfort_category(
    epw: EPW,
    st_hour: int = 0,
    end_hour: int = 23,
    density: bool = True,
    simplified: bool = False,
) -> pd.DataFrame:
    """
    Based on the best/worst conditions that could be envisaged in an EPWs
    location, determine what the upper and lower bounds for UTCI values might
    be for each hour of the year.

    Args:
        epw (EPW):
            An EPW object
        st_hour (float, optional):
            The start hour for any time-based filtering to apply. Defaults to 0.
        end_hour (float, optional):
            The end-hour for any time-based filtering to apply. Defaults to 23.
        density (bool, optional):
            Return proportion of time rather than number of hours. Defaults to True.
        simplified (bool, optional):
            Simplify comfort categories to use below/within/upper instead of
            discrete UTCI categories. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = feasible_utci_limits(epw, as_dataframe=True)
    df = categorise(df, simplified=simplified)

    if (st_hour < 0) or (end_hour < 0) or (st_hour > 23) or (end_hour > 23):
        raise ValueError("hours must be within the range 0-23.")
    if st_hour > end_hour:
        hours = [i for i in range(0, 24, 1) if ((i >= end_hour) or (i <= st_hour))]
    else:
        hours = [i for i in range(0, 24, 1) if ((i >= st_hour) and (i <= end_hour))]

    cats = ["Too cold", "Comfortable", "Too hot"] if simplified else UTCI_LABELS
    columns = pd.MultiIndex.from_product(
        [[i.capitalize() for i in cats], ["min", "max"]]
    )

    temps = []
    for col_label in ["min", "max"]:
        temp = df[col_label][df.index.hour.isin(hours)]
        temp = temp.groupby(temp.index.month).value_counts(normalize=density).unstack()
        temp.columns = pd.MultiIndex.from_product([temp.columns, [col_label]])
        temps.append(temp)
    temp = pd.concat(temps, axis=1).reindex(columns=columns).fillna(0)
    temp.index = [calendar.month_abbr[i] for i in temp.index]
    return temp


def feasible_comfort_temporal(
    epw: EPW,
    st_hour: float = 0,
    end_hour: float = 23,
    seasonality: Union[Callable, str] = "monthly",
    comfort_limits: Tuple[float] = (9, 26),
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

    Returns:
        pd.DataFrame:
            A summary table.
    """

    sx = [
        None,
        seasonality_from_day_length,
        seasonality_from_month,
        seasonality_from_temperature,
        "monthly",
    ]
    if seasonality not in sx:
        raise ValueError(f"seasonality must be one of {sx}")

    min_utci, max_utci = feasible_utci_limits(epw)

    utci_range = pd.concat([to_series(min_utci), to_series(max_utci)], axis=1).agg(
        ["min", "max"], axis=1
    )

    analysis_period = AnalysisPeriod(st_hour=st_hour, end_hour=end_hour)
    ap_bool = to_boolean(analysis_period)
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

    if (
        seasonality
        == seasonality_from_month  # pylint: disable=comparison-with-callable
    ):
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

    if (
        seasonality
        == seasonality_from_day_length  # pylint: disable=comparison-with-callable
    ):
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

    if (
        seasonality
        == seasonality_from_temperature  # pylint: disable=comparison-with-callable
    ):
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

    raise ValueError("How did you get here?")
