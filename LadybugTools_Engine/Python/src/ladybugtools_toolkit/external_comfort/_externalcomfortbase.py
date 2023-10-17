"""Base class for external comfort objects."""

from typing import Optional  # pylint: disable=E0401

import matplotlib.pyplot as plt
import pandas as pd
from ladybug.epw import AnalysisPeriod, HourlyContinuousCollection
from ladybug_comfort.collection.pmv import PMV
from ladybug_comfort.collection.utci import UTCI
from ladybug_geometry.geometry3d.pointvector import Point3D
from matplotlib.figure import Figure
from pydantic import (
    BaseModel,
    Field,  # pylint: disable=E0611
    root_validator,
    validator,
)
from ..bhom import CONSOLE_LOGGER, decorator_factory
from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, Categorical
from ..ladybug_extension.analysisperiod import describe_analysis_period
from ..ladybug_extension.datacollection import collection_to_series
from ..plot._heatmap import heatmap
from ..plot._utci import (
    utci_day_comfort_metrics,
    utci_heatmap,
    utci_heatmap_histogram,
    utci_histogram,
)
from ..plot.colormaps import (
    DBT_COLORMAP,
    MRT_COLORMAP,
    RH_COLORMAP,
    UTCI_DISTANCE_TO_COMFORTABLE,
    WS_COLORMAP,
)
from ._simulatebase import EnergyMaterial, EnergyMaterialVegetation, SimulationResult
from ._typologybase import Point3D, Typology
from .utci import distance_to_comfortable


class ExternalComfort(BaseModel):
    """_"""

    simulation_result: SimulationResult = Field(alias="SimulationResult")
    typology: Typology = Field(alias="Typology")

    dry_bulb_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="DryBulbTemperature"
    )
    relative_humidity: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="RelativeHumidity"
    )
    wind_speed: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="WindSpeed"
    )
    mean_radiant_temperature: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="MeanRadiantTemperature"
    )
    universal_thermal_climate_index: Optional[HourlyContinuousCollection] = Field(
        default=None, alias="UniversalThermalClimateIndex"
    )

    class Config:
        """_"""

        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        json_encoders = {
            Point3D: lambda v: v.to_dict(),
            EnergyMaterial: lambda v: v.to_dict(),
            EnergyMaterialVegetation: lambda v: v.to_dict(),
            HourlyContinuousCollection: lambda v: v.to_dict(),
        }

    @root_validator
    @classmethod
    def post_init_calculation(cls, values):  # pylint: disable=E0213
        """_"""

        _typology = values["typology"]
        _simulation_result = values["simulation_result"]
        _epw = values["simulation_result"].epw

        # run calculation and populate object with results if not already done
        if not values["dry_bulb_temperature"]:
            values["dry_bulb_temperature"] = _typology.dry_bulb_temperature(_epw)

        if not values["relative_humidity"]:
            values["relative_humidity"] = _typology.relative_humidity(_epw)

        if not values["wind_speed"]:
            values["wind_speed"] = _typology.wind_speed(_epw)

        if not values["mean_radiant_temperature"]:
            values["mean_radiant_temperature"] = _typology.mean_radiant_temperature(
                _simulation_result
            )

        if not values["universal_thermal_climate_index"]:
            values["universal_thermal_climate_index"] = UTCI(
                air_temperature=values["dry_bulb_temperature"],
                rel_humidity=values["relative_humidity"],
                rad_temperature=values["mean_radiant_temperature"],
                wind_speed=values["wind_speed"],
            ).universal_thermal_climate_index

        return values

    @validator(
        "dry_bulb_temperature",
        "relative_humidity",
        "wind_speed",
        "mean_radiant_temperature",
        "universal_thermal_climate_index",
        pre=True,
    )
    @classmethod
    def convert_dict_to_collection(cls, value: dict) -> object:  # pylint: disable=E0213
        """_"""
        if not isinstance(value, dict):
            return value
        if "type" not in value:
            return value
        if value["type"] == "HourlyContinuous":
            return HourlyContinuousCollection.from_dict(value)
        return value

    @decorator_factory()
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
        CONSOLE_LOGGER(
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

        return f"{self.simulation_result.description()}\n{self.typology.name}"

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

        return utci_heatmap(
            utci_collection=self.universal_thermal_climate_index,
            ax=ax,
            utci_categories=utci_categories,
            title=self.description(),
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

        return utci_histogram(
            utci_collection=self.universal_thermal_climate_index.filter_by_analysis_period(
                analysis_period
            ),
            ax=ax,
            utci_categories=utci_categories,
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
