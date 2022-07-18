# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any, Dict, List

# import numpy as np
# import pandas as pd
# from honeybee_energy.material.opaque import _EnergyMaterialOpaqueBase
# from ladybug.datacollection import HourlyContinuousCollection
# from ladybug.epw import EPW
# from ladybugtools_toolkit.external_comfort.encoder import Encoder
# from ladybugtools_toolkit.external_comfort.moisture.evaporative_cooling_effect_collection import (
#     evaporative_cooling_effect_collection,
# )
# from ladybugtools_toolkit.external_comfort.shelter.effective_wind_speed import (
#     effective_wind_speed,
# )
# from ladybugtools_toolkit.external_comfort.simulate.mean_radiant_temperature_result import (
#     MeanRadiantTemperatureResult,
# )
# from ladybugtools_toolkit.external_comfort.thermal_comfort.universal_thermal_climate_index import (
#     universal_thermal_climate_index,
# )
# from ladybugtools_toolkit.external_comfort.typology import Typology
# from ladybugtools_toolkit.ladybug_extension.datacollection.from_series import (
#     from_series,
# )
# from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
# from ladybugtools_toolkit.ladybug_extension.epw.to_dataframe import to_dataframe
# from ladybugtools_toolkit.ladybug_extension.helpers.decay_rate_smoother import (
#     decay_rate_smoother,
# )


# class ExternalComfort(MeanRadiantTemperatureResult):
#     """An object containing the results from an external comfort simulation applied to a
#         Typology containing shelters and evaporative cooling effects.

#     Args:
#         epw (EPW): An EPW object to be used for the mean radiant temperature simulation.
#         ground_material (_EnergyMaterialOpaqueBase): A material to use for the ground surface.
#         shade_material (_EnergyMaterialOpaqueBase): A material to use for the shade surface.
#         typology (Typology): A Typology object.
#         identifier (str, optional): A unique identifier for this case. If not provided,
#             then one will be created.

#     Returns:
#         ExternalComfort: An External Comfort object.
#     """

#     def __init__(
#         self,
#         epw: EPW,
#         ground_material: _EnergyMaterialOpaqueBase,
#         shade_material: _EnergyMaterialOpaqueBase,
#         typology: Typology,
#         identifier: str = None,
#     ) -> ExternalComfort:
#         super().__init__(epw, ground_material, shade_material, identifier)
#         self.typology = typology

#         # calculate properties
#         self._sky_exposure = self.typology.sky_exposure()
#         self._sun_exposure = self.typology.sun_exposure(
#             self.mean_radiant_temperature_result.epw
#         )

#         self.wind_speed = effective_wind_speed(
#             self.typology.shelters, self.mean_radiant_temperature_result.epw
#         )
#         (
#             self.dry_bulb_temperature,
#             self.relative_humidity,
#         ) = evaporative_cooling_effect_collection(
#             self.mean_radiant_temperature_result.epw,
#             self.typology.evaporative_cooling_effect,
#         )
#         self.mean_radiant_temperature = self._mean_radiant_temperature()
#         self.universal_thermal_climate_index = self._universal_thermal_climate_index()

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__} for {self.typology}"

#     def _mean_radiant_temperature(self) -> HourlyContinuousCollection:
#         """Return the effective mean radiant temperature for the given typology.

#         Returns:
#             HourlyContinuousCollection: An calculated mean radiant temperature based on the shelter configuration for the given typology.
#         """

#         shaded_mrt = to_series(
#             self.mean_radiant_temperature_result.shaded_mean_radiant_temperature
#         )
#         unshaded_mrt = to_series(
#             self.mean_radiant_temperature_result.unshaded_mean_radiant_temperature
#         )

#         daytime = np.array(
#             [
#                 True if i > 0 else False
#                 for i in self.mean_radiant_temperature_result.epw.global_horizontal_radiation
#             ]
#         )
#         mrts = []
#         for hour in range(8760):
#             if daytime[hour]:
#                 mrts.append(
#                     np.interp(
#                         self._sun_exposure[hour],
#                         [0, 1],
#                         [shaded_mrt[hour], unshaded_mrt[hour]],
#                     )
#                 )
#             else:
#                 mrts.append(
#                     np.interp(
#                         self._sky_exposure,
#                         [0, 1],
#                         [shaded_mrt[hour], unshaded_mrt[hour]],
#                     )
#                 )

#         # Fill any gaps where sun-visible/sun-occluded values are missing, and apply an exponentially weighted moving average to account for transition betwen shaded/unshaded periods.
#         mrt_series = pd.Series(
#             mrts, index=shaded_mrt.index, name=shaded_mrt.name
#         ).interpolate()

#         mrt_series = decay_rate_smoother(
#             mrt_series, difference_threshold=-10, transition_window=4, ewm_span=1.25
#         )

#         return from_series(mrt_series)

#     def _universal_thermal_climate_index(self) -> HourlyContinuousCollection:
#         return universal_thermal_climate_index(
#             self.dry_bulb_temperature,
#             self.relative_humidity,
#             self.mean_radiant_temperature,
#             self.wind_speed,
#         )

#     def to_dict(self) -> Dict[str, Any]:
#         """Return this object as a dictionary

#         Returns:
#             Dict: The dict representation of this object.
#         """

#         attributes = [
#             "dry_bulb_temperature",
#             "relative_humidity",
#             "wind_speed",
#             "mean_radiant_temperature",
#             "universal_thermal_climate_index",
#         ]

#         return {
#             **{
#                 attribute: getattr(self, attribute).to_dict()
#                 for attribute in attributes
#             },
#             **{"external_comfort": self.mean_radiant_temperature_result},
#         }

#     def to_json(self, file_path: str) -> Path:
#         """Return this object as a json file

#         Returns:
#             Path: The json file path.
#         """

#         file_path: Path = Path(file_path)
#         file_path.parent.mkdir(exist_ok=True, parents=True)

#         with open(file_path, "w", encoding="utf-8") as fp:
#             json.dump(self.to_dict(), fp, cls=Encoder, indent=4)

#         return file_path

#     def to_dataframe(
#         self, include_external_comfort_results: bool = True
#     ) -> pd.DataFrame:
#         """Create a dataframe from the typology results.

#         Args:
#             include_external_comfort_results (bool, optional): Whether to include the external
#                 comfort results in the dataframe. Defaults to True.

#         Returns:
#             pd.DataFrame: A dataframe containing the typology results.
#         """

#         attributes = [
#             "dry_bulb_temperature",
#             "relative_humidity",
#             "wind_speed",
#             "mean_radiant_temperature",
#             "universal_thermal_climate_index",
#         ]
#         series: List[pd.Series] = []
#         for attribute in attributes:
#             series.append(to_series(getattr(self, attribute)))
#         df = pd.concat(
#             series,
#             axis=1,
#             keys=[(self.typology.name, i) for i in attributes],
#         )

#         if include_external_comfort_results:
#             temp_df = to_dataframe(self.mean_radiant_temperature_result.epw)
#             temp_df.columns = [
#                 (Path(self.mean_radiant_temperature_result.epw.file_path).stem, i)
#                 for i in temp_df.columns
#             ]
#             df = pd.concat([df, temp_df], axis=1)

#         return df

#     # def plot_utci_day(self, month: int = 6, day: int = 21) -> Figure:
#     #     """Plot a single day UTCI and composite components

#     #     Args:
#     #         month (int, optional): The month to plot. Defaults to 6.
#     #         day (int, optional): The day to plot. Defaults to 21.

#     #     Returns:
#     #         Figure: A figure showing UTCI and component parts for the given day.
#     #     """
#     #     return plot_typology_day(
#     #         utci=to_series(self.universal_thermal_climate_index),
#     #         dbt=to_series(self.dry_bulb_temperature),
#     #         mrt=to_series(self.mean_radiant_temperature),
#     #         rh=to_series(self.relative_humidity),
#     #         ws=to_series(self.wind_speed),
#     #         month=month,
#     #         day=day,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #     )

#     # def plot_utci_heatmap(self) -> Figure:
#     #     """Create a heatmap showing the annual hourly UTCI values associated with this Typology.

#     #     Returns:
#     #         Figure: A matplotlib Figure object.
#     #     """

#     #     fig = timeseries_heatmap(
#     #         series=to_series(self.universal_thermal_climate_index),
#     #         cmap=UTCI_COLORMAP,
#     #         norm=UTCI_BOUNDARYNORM,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #     )

#     #     return fig

#     # def plot_utci_histogram(self) -> Figure:
#     #     """Create a histogram showing the annual hourly UTCI values associated with this Typology."""

#     #     fig = plot_utci_heatmap_histogram(
#     #         self.universal_thermal_climate_index,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #     )

#     #     return fig

#     # def plot_dbt_heatmap(self, vlims: List[float] = None) -> Figure:
#     #     """Create a heatmap showing the annual hourly DBT values associated with this Typology.

#     #     Args:
#     #         vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

#     #     Returns:
#     #         Figure: A matplotlib Figure object.
#     #     """

#     #     fig = timeseries_heatmap(
#     #         series=to_series(self.dry_bulb_temperature),
#     #         cmap=DBT_COLORMAP,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #         vlims=vlims,
#     #     )

#     #     return fig

#     # def plot_rh_heatmap(self, vlims: List[float] = None) -> Figure:
#     #     """Create a heatmap showing the annual hourly RH values associated with this Typology.

#     #     Args:
#     #         vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

#     #     Returns:
#     #         Figure: A matplotlib Figure object.
#     #     """

#     #     fig = timeseries_heatmap(
#     #         series=to_series(self.relative_humidity),
#     #         cmap=RH_COLORMAP,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #         vlims=vlims,
#     #     )

#     #     return fig

#     # def plot_ws_heatmap(self, vlims: List[float] = None) -> Figure:
#     #     """Create a heatmap showing the annual hourly WS values associated with this Typology.

#     #     Args:
#     #         vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

#     #     Returns:
#     #         Figure: A matplotlib Figure object.
#     #     """

#     #     fig = timeseries_heatmap(
#     #         series=to_series(self.wind_speed),
#     #         cmap=WS_COLORMAP,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #         vlims=vlims,
#     #     )

#     #     return fig

#     # def plot_mrt_heatmap(self, vlims: List[float] = None) -> Figure:
#     #     """Create a heatmap showing the annual hourly MRT values associated with this Typology.

#     #     Args:
#     #         vlims (List[float], optional): A list of two values to set the lower and upper limits of the colorbar. Defaults to None.

#     #     Returns:
#     #         Figure: A matplotlib Figure object.
#     #     """

#     #     fig = timeseries_heatmap(
#     #         series=to_series(self.mean_radiant_temperature),
#     #         cmap=MRT_COLORMAP,
#     #         title=f"{to_string(self.external_comfort_result.external_comfort.epw.location)}\n{self.typology.name}",
#     #         vlims=vlims,
#     #     )

#     #     return fig
