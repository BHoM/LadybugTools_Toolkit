import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

from typing import List
from ladybug.epw import EPW, HourlyContinuousCollection, Header, AnalysisPeriod
from ladybug.datatype.speed import WindSpeed
from ladybug.datatype.temperature import MeanRadiantTemperature
from ladybug_comfort.collection.utci import UTCI
import warnings, inspect

from external_comfort.openfield import Openfield
from external_comfort.evaporative_cooling import get_evaporative_cooled_dbt_rh
from ladybug_extension.datacollection import to_series, from_series
from ladybug_extension.sun import suns

class Typology:
    def __init__(
        self, 
        openfield: Openfield, 
        name: str = "Openfield", 
        evaporative_cooling_effectiveness: float = 0, 
        shelter_porosity: float = 1, 
        sheltered_azimuth_range: List[float] = [0, 0], 
        sheltered_altitude_range: List[float] = [0, 0],
    ):
        """Class for defining a specific external comfort typology, and calculating the resultant thermal comfort values.
        
        Args:
            openfield (Openfield): An Openfield object.
            name (str, optional): A string for the name of the typology. Defaults to "Openfield".
            evaporative_cooling_effectiveness (float, optional): A float between 0 and 1 for the effectiveness of the contextual evaporative cooling modifying air temperature. Defaults to 0.
            shelter_porosity (float, optional): A float between 0 and 1 for the transmissivity of the shade. Defaults to 1 representing full transmissivity.
            sheltered_azimuth_range (List[float], optional): A list of two floats between 0 and 360 for the azimuth range of the shelter. Defaults to [0, 0] representing no shelter.
            sheltered_altitude_range (List[float], optional): A list of two floats between -90 and 90 for the altitude range of the shelter. Defaults to [0, 0] representing no shelter.
        """
        self.name = name
        self.openfield = openfield
        self.evaporative_cooling_effectiveness = evaporative_cooling_effectiveness
        self.shelter_porosity = shelter_porosity
        self.sheltered_azimuth_range = sorted(sheltered_azimuth_range)
        self.sheltered_altitude_range = sorted(sheltered_altitude_range)

        if not ((0 <= self.evaporative_cooling_effectiveness <= 1) or isinstance(self.evaporative_cooling_effectiveness, (int, float))):
            raise ValueError(
                f"evaporative_cooling_effectiveness must be a number between 0 and 1"
            )

        if not (0 <= self.shelter_porosity <= 1):
            raise ValueError(f"shelter_porosity must be between 0 and 1")

        if (min(self.sheltered_azimuth_range) < 0) or (
            max(self.sheltered_azimuth_range) > 360
        ):
            raise ValueError(
                f"sheltered_azimuth_range must be values between 0 and 360"
            )

        if (min(self.sheltered_altitude_range) < 0) or (
            max(self.sheltered_altitude_range) > 90
        ):
            raise ValueError(
                f"sheltered_altitude_range must be values between 0 and 90"
            )

        if len(self.sheltered_azimuth_range) != 2:
            raise ValueError(f"sheltered_azimuth_range must be a list of length 2")

        if len(self.sheltered_altitude_range) != 2:
            raise ValueError(f"sheltered_altitude_range must be a list of length 2")

#     def _shading_mask(self) -> List[bool]:
#         return np.array(self._azimuth_mask()) & np.array(self._altitude_mask())
    
#     def _wind_speed(self, epw: EPW) -> HourlyContinuousCollection:
#         wind_speed_series = to_series(epw.wind_speed)

#         return HourlyContinuousCollection(
#             header=Header(
#                 analysis_period=AnalysisPeriod(), data_type=WindSpeed(), unit="m/s"
#             ),
#             values=wind_speed_series.where(self._azimuth_mask(), 0).values,
#         )
    
#     def _mean_radiant_temperature(self, openfield: Openfield) -> HourlyContinuousCollection:
#         shaded_series = to_series(openfield.shaded_mean_radiant_temperature)
#         unshaded_series = to_series(openfield.unshaded_mean_radiant_temperature)

#         diff = shaded_series - unshaded_series
#         adjustment = diff * (1 - self.shade_transmissivity)
#         partially_shaded_series = unshaded_series + adjustment

#         resultant_mrt = partially_shaded_series.where(
#             self._shading_mask(), openfield.unshaded_mean_radiant_temperature
#         )

#         return HourlyContinuousCollection(
#             header=Header(
#                 analysis_period=AnalysisPeriod(),
#                 data_type=MeanRadiantTemperature(),
#                 unit="C",
#             ),
#             values=resultant_mrt.values,
#         )
    
#     def universal_thermal_climate_index(self, openfield: Openfield) -> HourlyContinuousCollection:
#         # TODO - remove assertion when openfield is guaranteed to have MRT values
#         # assert openfield.shaded_mean_radiant_temperature is not None, \
#         #     "openfield.shaded_mean_radiant_temperature must be set"
#         # assert openfield.unshaded_mean_radiant_temperature is not None, \
#         #     "openfield.unshaded_mean_radiant_temperature must be set"
        
#         return UTCI(
#             air_temperature=self._dry_bulb_temperature(openfield.epw),
#             rel_humidity=self._relative_humidity(openfield.epw),
#             rad_temperature=self._mean_radiant_temperature(openfield),
#             wind_speed=self._wind_speed(openfield.epw),
#         ).universal_thermal_climate_index

    
    @property
    def description(self) -> str:
        """Return a text description of this external comfort typology."""

        descriptive_str = ""
        descriptive_str += self._shelter_description().capitalize()
        descriptive_str += f" with {self._wind_description()}"
        if self._evaporative_cooling_description() != "":
            descriptive_str += f", and {self._evaporative_cooling_description()}"
        return descriptive_str
    
    def _az_min(self) -> float:
        """Helper method to obtain the starting azimuth shelter angle."""
        return min(self.sheltered_azimuth_range)

    def _az_max(self) -> float:
        """Helper method to obtain the ending azimuth shelter angle."""
        return max(self.sheltered_azimuth_range)
    
    def _alt_min(self) -> float:
        """Helper method to obtain the starting altitude shelter angle."""
        return min(self.sheltered_altitude_range)
        
    def _alt_max(self) -> float:
        """Helper method to obtain the ending altitude shelter angle."""
        return max(self.sheltered_altitude_range)

    def _wind_speed_reduction_factor(self, altitude_threshold: float = 45) -> float:
        """Helper method to obtain a wind speed reduction factor, based on the proportion of vertical shelter below the altitude threshold.

        Args:
            altitude_threshold (float, optional): The altitude below which wind speed will be reduced. Defaults to 45.
        
        Returns:
            float: The wind speed reduction factor to be applied to the wind speed.
        """
        maximum_reduction_factor = 0.85
        low = self._alt_min()
        if low > altitude_threshold:
            return 1 * self.shelter_porosity
        high = min(altitude_threshold, self._alt_max())

        return np.interp(((high - low) / altitude_threshold), [0, 1], [1, maximum_reduction_factor]) * self.shelter_porosity

    def _shelter_description(self) -> str:
        """Return a text description of the the shelter component of this external comfort typology."""
        porosity_str = f"(with shelter porosity of {self.shelter_porosity:0.0%})"

        if (self.shelter_porosity == 1) or (self.sheltered_azimuth_range == [0, 0]) or (self._alt_max() <= 0):
            return "fully exposed (no shade or shelter)"
        
        if (self.sheltered_azimuth_range == [0, 360]) and (self.sheltered_altitude_range == [0, 0]):
            return f"sheltered to all sides {porosity_str}, and open to the sky"
        
        if (self.sheltered_azimuth_range == [0, 360]):
            return f"sheltered to all sides, and between altitudes of {self._alt_min()}° and {self._alt_max()}° {porosity_str}"
        
        return f"sheltered between {self._az_min()}° and {self._az_max()}° from North, and between altitudes of {self._alt_min()}° and {self._alt_max()}° {porosity_str}"

    def _wind_description(self) -> str:
        """Return a text description of the the wind componernt of this external comfort typology."""
        if (self.shelter_porosity == 1) or (self.sheltered_azimuth_range == [0, 0]) or (self._alt_max() <= 0):
            return "wind speed per weatherfile"
        
        if self.shelter_porosity == 0:
            return "wind speed removed from sheltered directions"
        else:
            return f"wind speed reduced by {1 - self.shelter_porosity:0.0%} from sheltered directions"
    
    def _evaporative_cooling_description(self) -> str:
        """Return a text description of the the evaporative cooling componernt of this external comfort typology."""
        if self.evaporative_cooling_effectiveness != 0:
            return f"{self.evaporative_cooling_effectiveness:0.0%} effective evaporative cooling"
        else:
            return ""
    
    def __str__(self) -> str:
        return f"{self.name} - {self.description}"

    def _wind_speed(self) -> HourlyContinuousCollection:
        """Adjust wind_speed based on shelter positioning and porosity."""
        if (self.shelter_porosity == 1) or (self.sheltered_azimuth_range == [0, 0]) or (self._alt_max() <= 0):
            return self.openfield.epw.wind_speed
        else:
            wd = to_series(self.openfield.epw.wind_direction)
            ws = to_series(self.openfield.epw.wind_speed)
            sheltered_mask = wd.between(self._az_min(), self._az_max()).values
            adjustment_factor = self._wind_speed_reduction_factor()
            ws_adjusted = ws.where(~sheltered_mask, ws * adjustment_factor)
        return from_series(ws_adjusted)

    def _dry_bulb_temperature(self) -> HourlyContinuousCollection:
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["dry_bulb_temperature"]
    
    def _relative_humidity(self) -> HourlyContinuousCollection:
        return get_evaporative_cooled_dbt_rh(
            self.openfield.epw, self.evaporative_cooling_effectiveness
        )["relative_humidity"]

    def _shadedness_factor(self) -> List[float]:
        """Return a list of values which can be used to interpolate between shaded/unshaded MRT values to provide an approximate point-in-time MRT."""
        
        # get the proportion of the sky covered by a shade, so that overnight effects can be approximated by adjusting between shaded/unsahded per fraction when sun.alt < 0
        radii = 1
        lat1 = self._az_min()
        lat2 = self._az_max()
        lon1 = self._alt_min()
        lon2 = self._alt_max()
        sky_coverage_area = (np.pi/180) * radii**2 * abs(np.sin(lat1)-np.sin(lat2)) * abs(lon1-lon2)
        
        height = 1.2
        
        sky_area = 2 * np.pi * (1 + (np.sqrt(height ** 2 + 2 * radii * height) / radii * height))
        sky_sheltered_proportion = sky_coverage_area / sky_area

        # print(sky_coverage_area, sky_area, sky_sheltered_proportion)

        # TODO - Vary between shaded/unshaded based on sun proximity to shade centroid??/coverage?? and include shelter porosity!

        sun_objects = suns(self.openfield.epw)

        return np.ones(8760)

    def _mean_radiant_temperature(self) -> HourlyContinuousCollection:
        shaded_series = to_series(self.openfield.shaded_mean_radiant_temperature)
        unshaded_series = to_series(self.openfield.unshaded_mean_radiant_temperature)

        mrt_values = [np.interp(i, [0, 1], [unshaded_series[n], shaded_series[n]]) for n, i in enumerate(self._shadedness_factor())]

        resultant_mrt = pd.Series(mrt_values, name=shaded_series.name, index=shaded_series.index)

        return from_series(resultant_mrt)


if __name__ == "__main__":
    from external_comfort.material import MATERIALS

    epw = EPW(r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\test\GBR_London.Gatwick.037760_IWEC.epw")
    ground_material = MATERIALS["CONCRETE_LIGHTWEIGHT"]
    shade_material = MATERIALS["FABRIC"]
    openfield = Openfield(epw, ground_material, shade_material, True)
    typology = Typology(
        openfield, 
        name="Example", 
        evaporative_cooling_effectiveness=0, 
        sheltered_azimuth_range=[0, 360], 
        sheltered_altitude_range=[0, 90], 
        shelter_porosity=0.5
    )

    # print(typology._wind_speed().average)
    # print(typology._dry_bulb_temperature().average)
    # print(typology._relative_humidity().average)
    print(typology._mean_radiant_temperature())
