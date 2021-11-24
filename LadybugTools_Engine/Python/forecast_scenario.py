from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import fortranformat as ff
import pandas as pd
from ladybug.location import Location
from scipy import spatial

from enums import EmissionsScenario, ForecastYear


def load_points_kml(kml_path: Path) -> List[List[float]]:
    points = []
    with open(kml_path, "r") as f:
        for line in f:
            if line.startswith("<coordinates>"):
                line = (
                    line.replace("<coordinates>", "")
                    .replace("</coordinates>", "")
                    .strip()
                )
                lat, long = line.split(",")
                points.append([float(lat), float(long)])
    return points


def load_variable_dif(file_path: Path) -> List[List[float]]:
    with open(file_path, "r") as fp:
        data = fp.readlines()
    starts = [n for n, i in enumerate(data) if i.startswith("IPCC")]
    starts += [len(data)]
    header_rows = 6
    indices = list(zip(starts, starts[1:]))

    config_row = data[indices[0][0] : indices[0][1] + header_rows]
    n_values = int(config_row[-1].split()[0])
    format = config_row[-1].split()[3]

    reader = ff.FortranRecordReader(format)

    values = []
    for x, y in indices:
        temp = []
        for row in data[x:y][header_rows:]:
            temp.extend(reader.read(row))
        values.append(temp[:n_values])

    return values


def nearest_n_point_indices(
    points: List[List[float]], location: Location, n: int
) -> List[int]:
    _, nearest_point_indices = spatial.KDTree(points).query(
        [location.latitude, location.longitude], k=n
    )
    return nearest_point_indices


def construct_file_path(
    root_directory: Path,
    climate_variable: str,
    emissions_scenario: EmissionsScenario,
    forecast_year: ForecastYear,
) -> Path:
    file_path = (
        root_directory
        / f"HADCM3_{emissions_scenario.value}_{climate_variable}_{forecast_year.value}.dif"
    )

    if file_path.exists():
        return file_path
    else:
        raise FileNotFoundError(
            f"It doesn't seem as though a dataset is available for {file_path.name}."
        )


class TranslationFactors:
    def __init__(
        self,
        location: Location,
        emissions_scenario: EmissionsScenario,
        forecast_year: ForecastYear,
        DSWF: pd.Series = None,
        MSLP: pd.Series = None,
        PREC: pd.Series = None,
        RHUM: pd.Series = None,
        TCLW: pd.Series = None,
        TEMP: pd.Series = None,
        TMAX: pd.Series = None,
        TMIN: pd.Series = None,
        WIND: pd.Series = None,
    ):
        self.location = location
        self.emissions_scenario = emissions_scenario
        self.forecast_year = forecast_year
        self.DSWF = DSWF
        self.MSLP = MSLP
        self.PREC = PREC
        self.RHUM = RHUM
        self.TCLW = TCLW
        self.TEMP = TEMP
        self.TMAX = TMAX
        self.TMIN = TMIN
        self.WIND = WIND

    def __str__(self) -> str:
        return f"{self.location.city}-{self.emissions_scenario}-{self.forecast_year}"

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"


class ForecastScenario:
    def __init__(
        self, emissions_scenario: EmissionsScenario, forecast_year: ForecastYear
    ):

        self.emissions_scenario = emissions_scenario
        self.forecast_year = forecast_year

        self._root_directory = (
            Path(__file__).parent / "datasets"
        )
        self._month_idx = pd.date_range("2021-01-01", freq="MS", periods=12)
        self._year_idx = pd.date_range("2021-01-01 00:30:00", freq="60T", periods=8760)

        def __setter(
            obj: object, var: str, dir: Path, es: EmissionsScenario, fy: ForecastYear
        ):
            setattr(obj, var, load_variable_dif(construct_file_path(dir, var, es, fy)))

        results = []
        #print(f"Loading {self} datasets")
        with ThreadPoolExecutor() as executor:
            for var in [
                "DSWF",
                "MSLP",
                "PREC",
                "RHUM",
                "TCLW",
                "TEMP",
                "TMIN",
                "TMAX",
                "WIND",
            ]:
                results.append(
                    executor.submit(
                        __setter,
                        self,
                        var,
                        self._root_directory,
                        self.emissions_scenario,
                        self.forecast_year,
                    )
                )

        self._points = load_points_kml(self._root_directory / "HADCM3_grid_centre.kml")
        self._wind_points = load_points_kml(
            self._root_directory / "HADCM3_grid_WIND_centre.kml"
        )

    def get_translation_factors(self, location: Location) -> TranslationFactors:
        """
        Get the translation factors for a given location.
        """
        nearest_point_indices = nearest_n_point_indices(self._points, location, n=4)
        nearest_wind_point_indices = nearest_n_point_indices(
            self._wind_points, location, n=4
        )

        def __mp(self, obj: TranslationFactors, var: str) -> pd.Series:
            if var == "WIND":
                setattr(
                    obj,
                    var,
                    pd.DataFrame(getattr(self, var), index=self._month_idx)
                    .reindex(self._year_idx, method="ffill")
                    .iloc[:, nearest_wind_point_indices]
                    .mean(axis=1),
                )
            else:
                setattr(
                    obj,
                    var,
                    pd.DataFrame(getattr(self, var), index=self._month_idx)
                    .reindex(self._year_idx, method="ffill")
                    .iloc[:, nearest_point_indices]
                    .mean(axis=1),
                )

        translations = TranslationFactors(
            location, self.emissions_scenario, self.forecast_year
        )
        results = []
        #print(f"Calculating translation factors for {self}")
        with ThreadPoolExecutor() as executor:
            for var in [
                "DSWF",
                "MSLP",
                "PREC",
                "RHUM",
                "TCLW",
                "TEMP",
                "TMIN",
                "TMAX",
                "WIND",
            ]:
                results.append(
                    executor.submit(
                        __mp,
                        self,
                        translations,
                        var,
                    )
                )

        return translations

    def __str__(self) -> str:
        return f"{self.emissions_scenario}-{self.forecast_year} forecast"

    def __repr__(self) -> str:
        return f"{__class__.__name__}[{self}]"
