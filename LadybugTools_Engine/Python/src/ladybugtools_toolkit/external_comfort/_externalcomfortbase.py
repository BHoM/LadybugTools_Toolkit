"""Base class for external comfort objects."""
# pylint: disable=E0401
import json
from dataclasses import dataclass
from pathlib import Path

# pylint: enable=E0401
from caseconverter import pascalcase
import matplotlib.pyplot as plt
import pandas as pd
from ladybug.epw import AnalysisPeriod, HourlyContinuousCollection
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

from ..bhom.logging import CONSOLE_LOGGER
from ..bhom.to_bhom import hourlycontinuouscollection_to_bhom
from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical
from ..helpers import convert_keys_to_snake_case
from ..ladybug_extension.analysisperiod import describe_analysis_period
from ..ladybug_extension.datacollection import collection_to_series
from python_toolkit.plot.heatmap import heatmap
from ..plot._utci import utci_day_comfort_metrics, utci_heatmap_histogram
from ..plot.colormaps import (
    DBT_COLORMAP,
    MRT_COLORMAP,
    RH_COLORMAP,
    UTCI_DISTANCE_TO_COMFORTABLE,
    WS_COLORMAP,
)
from ._simulatebase import SimulationResult
from .typology import Typology, Typologies
from .utci import distance_to_comfortable
from ..ladybug_extension.header import header_from_string


_ATTRIBUTES = [
    "dry_bulb_temperature",
    "relative_humidity",
    "wind_speed",
    "mean_radiant_temperature",
    "universal_thermal_climate_index",
]


@dataclass(init=True, repr=True, eq=True)
class ExternalComfort:
    """_"""

    simulation_result: SimulationResult
    typology: Typology

    dry_bulb_temperature: HourlyContinuousCollection = None
    relative_humidity: HourlyContinuousCollection = None
    wind_speed: HourlyContinuousCollection = None
    mean_radiant_temperature: HourlyContinuousCollection = None
    universal_thermal_climate_index: HourlyContinuousCollection = None

    def __post_init__(self):
        """_"""

        # validation
        if not isinstance(self.simulation_result, SimulationResult):
            raise ValueError(
                "simulation_result must be an instance of SimulationResult."
            )

        if isinstance(self.typology, Typologies):
            self.typology = self.typology.value
        if not isinstance(self.typology, Typology):
            raise ValueError("typology must be an instance of Typology.")

        for attr in _ATTRIBUTES:
            if not isinstance(
                getattr(self, attr), (HourlyContinuousCollection, type(None))
            ):
                raise ValueError(
                    f"{attr} must be an instance of HourlyContinuousCollection or None."
                )

        CONSOLE_LOGGER.info(
            f"Processing {self.simulation_result.model.identifier} - {self.typology.identifier}"
        )

        # run calculation and populate object with results if not already done
        _epw = self.simulation_result.epw

        if not getattr(self, "dry_bulb_temperature"):
            setattr(
                self, "dry_bulb_temperature", self.typology.dry_bulb_temperature(_epw)
            )

        if not getattr(self, "relative_humidity"):
            setattr(self, "relative_humidity", self.typology.relative_humidity(_epw))

        if not getattr(self, "wind_speed"):
            setattr(self, "wind_speed", self.typology.wind_speed(_epw))

        if not getattr(self, "mean_radiant_temperature"):
            setattr(
                self,
                "mean_radiant_temperature",
                self.typology.mean_radiant_temperature(self.simulation_result),
            )

        if not getattr(self, "universal_thermal_climate_index"):
            setattr(
                self,
                "universal_thermal_climate_index",
                UTCI(
                    air_temperature=self.dry_bulb_temperature,
                    rel_humidity=self.relative_humidity,
                    rad_temperature=self.mean_radiant_temperature,
                    wind_speed=self.wind_speed,
                ).universal_thermal_climate_index,
            )

        # add some accessors for collections as series
        for attr in _ATTRIBUTES:
            setattr(self, f"{attr}_series", collection_to_series(getattr(self, attr)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.simulation_result}, {self.typology})"

    def to_dict(self) -> str:
        """Convert this object to a dictionary."""
        attr_dict = {}
        for attr in _ATTRIBUTES:
            if getattr(self, attr):
                attr_dict[attr] = getattr(self, attr).to_dict()

        d = {
            **{
                "type": "ExternalComfort",
                "simulation_result": self.simulation_result.to_dict(),
                "typology": self.typology.to_dict(),
            },
            **attr_dict,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExternalComfort":
        """Create a dictionary from this object."""
        if isinstance(d["simulation_result"], dict):
            d["simulation_result"] = SimulationResult.from_dict(d["simulation_result"])

        if isinstance(d["typology"], dict):
            d["typology"] = Typology.from_dict(d["typology"])

        for attr in _ATTRIBUTES:
            if d.get(attr, None):
                if isinstance(d[attr], dict):
                    d[attr] = HourlyContinuousCollection.from_dict(d[attr])
                else:
                    d[attr] = None
            else:
                d[attr] = None

        return cls(
            simulation_result=d["simulation_result"],
            typology=d["typology"],
            dry_bulb_temperature=d["dry_bulb_temperature"],
            relative_humidity=d["relative_humidity"],
            wind_speed=d["wind_speed"],
            mean_radiant_temperature=d["mean_radiant_temperature"],
            universal_thermal_climate_index=d["universal_thermal_climate_index"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "SimulationResult":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Write this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "SimulationResult":
        """Create this object from a JSON file."""

        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    def standard_effective_temperature(
        self,
        met_rate: float = 1.1,
        clo_value: float = 0.7,
        return_comfort_obj: bool = False,
    ) -> PMV | HourlyContinuousCollection:
        """Return the predicted mean vote for the given typology
            following a simulation of the collections necessary to calculate
            this.

        Args:
            met_rate (float, optional):
                Metabolic rate value.
            clo_value (float, optional):
                Clothing value.
            return_comfort_obj (bool, optional):
                Set to True to return the PMV comfort object instead of the
                data collection.

        Returns:
            [PMV, HourlyContinuousCollection]:
                The resultant PMV for this Typology, in the given
                SimulationResult location.
        """
        CONSOLE_LOGGER.info(
            f"{self.__class__.__name__} - calculating standard effective temperature"
        )
        pmv = PMV(
            air_temperature=self.dry_bulb_temperature,
            rel_humidity=self.relative_humidity,
            rad_temperature=self.mean_radiant_temperature,
            air_speed=self.wind_speed,
            met_rate=met_rate,
            clo_value=clo_value,
        )

        if return_comfort_obj:
            return pmv

        return pmv.standard_effective_temperature

    def description(self) -> str:
        """Return a description of this external comfort object."""
        return (
            f"{self.simulation_result.description(include_shade_material=bool(self.typology.shelters))}"
            f"\n{self.typology.identifier}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return a dataframe of all data for this typology."""

        simulation_result_df = self.simulation_result.to_dataframe()
        obj_df = pd.concat(
            [
                pd.concat(
                    [
                        collection_to_series(self.dry_bulb_temperature),
                        collection_to_series(self.relative_humidity),
                        collection_to_series(self.mean_radiant_temperature),
                        collection_to_series(self.wind_speed),
                        pd.Series(
                            self.typology.evaporative_cooling_effect,
                            index=simulation_result_df.index,
                            name="Evaporative Cooling Effect (0-1)",
                        ),
                        pd.Series(
                            self.typology.radiant_temperature_adjustment,
                            index=simulation_result_df.index,
                            name="Radiant Temperature Adjustment (C)",
                        ),
                        collection_to_series(self.universal_thermal_climate_index),
                    ],
                    axis=1,
                )
            ],
            axis=1,
            keys=[self.__class__.__name__],
        )

        return pd.concat([simulation_result_df, obj_df], axis=1)

    def plot_utci_day_comfort_metrics(
        self, ax: plt.Axes = None, month: int = 3, day: int = 21
    ) -> plt.Axes:
        """Plot a single day UTCI and composite components

        Args:
            ax (plt.Axes, optional):
                A matplotlib Axes object to plot on. Defaults to None.
            month (int, optional):
                The month to plot. Defaults to 3.
            day (int, optional):
                The day to plot. Defaults to 21.

        Returns:
            Axes: A figure showing UTCI and component parts for the given day.
        """

        return utci_day_comfort_metrics(
            utci=collection_to_series(self.universal_thermal_climate_index),
            dbt=collection_to_series(self.dry_bulb_temperature),
            mrt=collection_to_series(self.mean_radiant_temperature),
            rh=collection_to_series(self.relative_humidity),
            ws=collection_to_series(self.wind_speed),
            ax=ax,
            month=month,
            day=day,
            title=self.description(),
        )

    def plot_utci_heatmap(
        self,
        ax: plt.Axes = None,
        utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES,
    ) -> plt.Axes:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Args:
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. Defaults to None.
            utci_categories (Categorical, optional): The UTCI categories to use. Defaults to
                UTCI_DEFAULT_CATEGORIES.

        Returns:
            plt.Axes: A matplotlib Axes object.
        """

        return utci_categories.annual_heatmap(
            series=collection_to_series(self.universal_thermal_climate_index),
            ax=ax,
            title=self.description(),
        )

    def walkability_time_limits(self):

        """Calculate walkability time limiits
        Returns:
            HourlyContinuousCollection: An object containing walkability values for each hour. 
        """
        csv_file = r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\src\data\walkability.csv"
        df = pd.read_csv(csv_file)
        walkability_vals = []

        for utci_val in self.universal_thermal_climate_index.values:
            utci_int = int(round(utci_val,0))
            utci_str = str(utci_int)

            val = df.loc[df["UTCI temperature"] == utci_str, 'Time within no thermal stress and moderate thermal stress bands (mins)'].values
            if utci_int < 20:
                val = df.loc[df["UTCI temperature"] == "defaultlower", 'Time within no thermal stress and moderate thermal stress bands (mins)'].values
            elif utci_int > 60:
                val = df.loc[df["UTCI temperature"] == "defaultupper", 'Time within no thermal stress and moderate thermal stress bands (mins)'].values

            walkability_vals.append(val[0])

        header = header_from_string("Walkability Time (minutes)")

        return HourlyContinuousCollection(header=header, values=walkability_vals)

    def plot_walkability_heatmap( 
            self,
            ax: plt.Axes = None,
            **kwargs
        ) -> plt.Axes:
            
        """Create a walkability heatmap showing 
        Args:
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. Defaults to None.
            utci_categories (Categorical, optional): The UTCI categories to use. Defaults to
                UTCI_DEFAULT_CATEGORIES.
        Returns:
            plt.Axes: A matplotlib Axes object.
        """

        cmap_name = "Walkability colours"
        cmap_colours = ["red","yellow","white"]
        colourmap = LinearSegmentedColormap.from_list(cmap_name, cmap_colours, N=100)

        return heatmap(
            series = collection_to_series(self.walkability_time_limits()),
            ax=ax,
            cmap = colourmap,
            vmin=0,
            vmax=15,
            title=kwargs.pop("title", self.description()),
            **kwargs
        )

    def plot_utci_heatmap_histogram(
        self, utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES, **kwargs
    ) -> plt.Figure:
        """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

        Args:
            utci_categories (Categorical, optional): The UTCI categories to use. Defaults to
                UTCI_DEFAULT_CATEGORIES.
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.
        Returns:
            Figure: A matplotlib Figure object.
        """

        return utci_heatmap_histogram(
            utci_collection=self.universal_thermal_climate_index,
            utci_categories=utci_categories,
            title=self.description(),
            **kwargs,
        )

    def plot_utci_histogram(
        self,
        ax: plt.Axes = None,
        utci_categories: Categorical = UTCI_DEFAULT_CATEGORIES,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        title: str = None,
        **kwargs,
    ) -> plt.Axes:
        """Create a histogram showing the annual hourly UTCI values associated with this Typology.

        Args:
            ax (plt.Axes, optional):
                A matplotlib Axes object to plot on. Defaults to None.
            utci_categories (Categorical, optional):
                The UTCI categories to use. Defaults to UTCI_DEFAULT_CATEGORIES.
            analysis_period (AnalysisPeriod, optional):
                The analysis period to filter the results by. Defaults to AnalysisPeriod().
            title (str, optional):
                The title to use for the plot. Defaults to None which generates a title from the
                Typology and AnalysisPeriod.
            **kwargs:
                Additional keyword arguments to pass to the histogram function.
        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        if title is None:
            title = (
                f"{self.description()} ({describe_analysis_period(analysis_period)})"
            )

        return utci_categories.annual_monthly_histogram(
            series=collection_to_series(
                self.universal_thermal_climate_index.filter_by_analysis_period(
                    analysis_period
                )
            ),
            ax=ax,
            title=title,
            **kwargs,
        )

    def plot_utci_distance_to_comfortable(
        self,
        ax: plt.Axes = None,
        comfort_thresholds: tuple[float] = (9, 26),
        distance_to_comfort_band_centroid: bool = True,
    ) -> Figure:
        """Create a heatmap showing the "distance" in C from the "no thermal stress" UTCI comfort
            band.

        Args:
            ax (plt.Axes, optional): A matplotlib Axes object to plot on. Defaults to None.
            comfort_thresholds (List[float], optional): The comfortable band of UTCI temperatures.
                Defaults to [9, 26].
            distance_to_comfort_band_centroid (bool, optional): Set to True to calculate the
                distance to the centroid of the comfort band. Defaults to True.

        Returns:
            Axes: A matplotlib Axes object.
        """

        new_collection = distance_to_comfortable(
            utci_value=self.universal_thermal_climate_index,
            comfort_thresholds=comfort_thresholds,
            distance_to_comfort_band_centroid=distance_to_comfort_band_centroid,
        )

        return heatmap(
            collection_to_series(new_collection),
            ax=ax,
            cmap=UTCI_DISTANCE_TO_COMFORTABLE,
            title=f"{self.description()}\nDistance to comfortable",
            vmin=-10,
            vmax=10,
        )

    def plot_dbt_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly DBT values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.dry_bulb_temperature),
            cmap=DBT_COLORMAP,
            title=self.description(),
            **kwargs,
        )

    def plot_rh_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly RH values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.relative_humidity),
            cmap=RH_COLORMAP,
            title=self.description(),
            **kwargs,
        )

    def plot_ws_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly WS values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.wind_speed),
            cmap=WS_COLORMAP,
            title=self.description(),
            **kwargs,
        )

    def plot_mrt_heatmap(self, **kwargs) -> plt.Axes:
        """Create a heatmap showing the annual hourly MRT values associated with this Typology.

        Args:
            **kwargs:
                Additional keyword arguments to pass to the heatmap function.

        Returns:
            Figure: A matplotlib Figure object.
        """

        return heatmap(
            series=collection_to_series(self.mean_radiant_temperature),
            cmap=MRT_COLORMAP,
            title=self.description(),
            **kwargs,
        )
