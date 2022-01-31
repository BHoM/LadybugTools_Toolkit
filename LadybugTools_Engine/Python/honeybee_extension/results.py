from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ladybug.sql import SQLiteResult
from ladybug_extension.datacollection import to_series


class Results:
    def __init__(self, results_directory: Path):
        self.results_directory = (
            results_directory
            if isinstance(results_directory, Path)
            else Path(results_directory)
        )

        self.points = None
        self.energy = None
        self.daylight_factor = None
        self.sky_view = None
        self.direct_sun_hours = None

        self.annual_daylight_direct = None
        self.annual_daylight_total = None
        self.annual_irradiance_direct = None
        self.annual_irradiance_total = None

        self.daylight_autonomy = None
        self.continuous_daylight_autonomy = None
        self.useful_daylight_illuminance_lower = None
        self.useful_daylight_illuminance = None
        self.useful_daylight_illuminance_upper = None

        self.load_all()

    def __str__(self):
        return f"Honeybee results for {self.results_directory.as_posix()}"

    def load_energy(self) -> Results:
        """Add results from Energyplus simulation to the Results object."""
        # Find the SQL results file
        try:
            sql_path = list(self.results_directory.glob("**/*.sql"))[0]
        except IndexError:
            return self
        hdf_path = self.results_directory / "energy.hdf"
        try:
            print(f"Loading Energyplus results from {hdf_path}")
            self.energy = pd.read_hdf(hdf_path, "df")
        except FileNotFoundError:
            print(f"Loading Energyplus results from {sql_path}")
            self.energy = _load_sql(sql_path)
        return self

    def load_sky_view(self) -> Results:
        """Add results from Sky View simulation to the Results object."""
        res_path = self.results_directory / "sky_view" / "results"
        hdf_path = self.results_directory / "sky_view.hdf"
        try:
            self.sky_view = pd.read_hdf(hdf_path, "df")
        except FileNotFoundError:
            serieses = []
            for file in (res_path).glob("*.res"):
                serieses.append(_load_res(file))
            self.sky_view = pd.concat(serieses, axis=1)

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (self.results_directory / "sky_view" / "model" / "grid").glob(
                "*.pts"
            ):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_daylight_factor(self) -> Results:
        """Add results from Daylight Factor simulation to the Results object."""
        try:
            self.daylight_factor = pd.read_hdf(
                self.results_directory / "daylight_factor.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (self.results_directory / "daylight_factor" / "results").glob(
                "*.res"
            ):
                serieses.append(_load_res(file))
            self.daylight_factor = pd.concat(serieses, axis=1)

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (
                self.results_directory / "daylight_factor" / "model" / "grid"
            ).glob("*.pts"):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_direct_sun_hours(self) -> Results:
        """Add results from Direct Sun Hours simulation to the Results object."""
        try:
            self.direct_sun_hours = pd.read_hdf(
                self.results_directory / "direct_sun_hours.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "direct_sun_hours" / "results" / "cumulative"
            ).glob("*.res"):
                serieses.append(_load_res(file))
            self.direct_sun_hours = pd.concat(serieses, axis=1)

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (
                self.results_directory / "direct_sun_hours" / "model" / "grid"
            ).glob("*.pts"):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_annual_daylight(self) -> Results:
        """Add results from annual-daylight simulation to the Results object."""

        try:
            self.annual_daylight_direct = pd.read_hdf(
                self.results_directory / "annual_daylight_direct.hdf", "df"
            )
            self.annual_daylight_total = pd.read_hdf(
                self.results_directory / "annual_daylight_total.hdf", "df"
            )
        except FileNotFoundError:
            try:
                for _type in ["direct", "total"]:
                    frames = []
                    for file in (
                        self.results_directory / "annual_daylight" / "results" / _type
                    ).glob("*.ill"):
                        print(f"Loading {file}")
                        frames.append(_load_ill(file))
                        setattr(
                            self, f"annual_daylight_{_type}", pd.concat(frames, axis=1)
                        )
            except ValueError as e:
                raise UserWarning(e)

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (
                self.results_directory / "annual_daylight" / "model" / "grid"
            ).glob("*.pts"):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_annual_irradiance(self) -> Results:
        """Add results from annual-irradiance simulation to the Results object."""

        try:
            self.annual_irradiance_direct = pd.read_hdf(
                self.results_directory / "annual_irradiance_direct.hdf", "df"
            )
            self.annual_irradiance_total = pd.read_hdf(
                self.results_directory / "annual_irradiance_total.hdf", "df"
            )
        except FileNotFoundError:
            try:
                for _type in ["direct", "total"]:
                    frames = []
                    for file in (
                        self.results_directory / "annual_irradiance" / "results" / _type
                    ).glob("*.ill"):
                        print(f"Loading {file}")
                        frames.append(_load_ill(file))
                        setattr(
                            self,
                            f"annual_irradiance_{_type}",
                            pd.concat(frames, axis=1),
                        )
            except ValueError as e:
                raise UserWarning(f"{e}")

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (
                self.results_directory / "annual_irradiance" / "model" / "grid"
            ).glob("*.pts"):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_climate_based_daylight_metrics(self) -> Results:
        try:
            self.daylight_autonomy = pd.read_hdf(
                self.results_directory / "daylight_autonomy.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "annual_daylight" / "metrics" / "da"
            ).glob("*.da"):
                serieses.append(_load_res(file))
            self.daylight_autonomy = pd.concat(serieses, axis=1)

        try:
            self.continuous_daylight_autonomy = pd.read_hdf(
                self.results_directory / "continuous_daylight_autonomy.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "annual_daylight" / "metrics" / "cda"
            ).glob("*.cda"):
                serieses.append(_load_res(file))
            self.continuous_daylight_autonomy = pd.concat(serieses, axis=1)

        try:
            self.useful_daylight_illuminance_lower = pd.read_hdf(
                self.results_directory / "useful_daylight_illuminance_lower.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "annual_daylight" / "metrics" / "udi_lower"
            ).glob("*.udi"):
                serieses.append(_load_res(file))
            self.useful_daylight_illuminance_lower = pd.concat(serieses, axis=1)

        try:
            self.useful_daylight_illuminance = pd.read_hdf(
                self.results_directory / "useful_daylight_illuminance.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "annual_daylight" / "metrics" / "udi"
            ).glob("*.udi"):
                serieses.append(_load_res(file))
            self.useful_daylight_illuminance = pd.concat(serieses, axis=1)

        try:
            self.useful_daylight_illuminance_upper = pd.read_hdf(
                self.results_directory / "useful_daylight_illuminance_upper.hdf", "df"
            )
        except FileNotFoundError:
            serieses = []
            for file in (
                self.results_directory / "annual_daylight" / "metrics" / "udi_upper"
            ).glob("*.udi"):
                serieses.append(_load_res(file))
            self.useful_daylight_illuminance_upper = pd.concat(serieses, axis=1)

        try:
            self.points = pd.read_hdf(self.results_directory / "points.hdf", "df")
        except FileNotFoundError:
            pointses = []
            for file in (
                self.results_directory / "annual_daylight" / "model" / "grid"
            ).glob("*.pts"):
                pointses.append(_load_pts(file))
            self.points = pd.concat(pointses, axis=1)

        return self

    def load_all(self) -> Results:
        try:
            self.load_energy()
        except Exception as e:
            raise e

        try:
            self.load_sky_view()
        except Exception:
            pass

        try:
            self.load_direct_sun_hours()
        except Exception:
            pass

        try:
            self.load_daylight_factor()
        except Exception:
            pass

        try:
            self.load_annual_daylight()
        except Exception:
            pass

        try:
            self.load_annual_irradiance()
        except Exception:
            pass

        try:
            self.load_climate_based_daylight_metrics()
        except Exception:
            pass

        return self

    def to_hdf(self) -> None:
        """Save all loaded data into an HDF file for easier reading later!"""
        if self.energy is not None:
            self.energy.to_hdf(
                self.results_directory / "energy.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.sky_view is not None:
            self.sky_view.to_hdf(
                self.results_directory / "sky_view.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.direct_sun_hours is not None:
            self.direct_sun_hours.to_hdf(
                self.results_directory / "direct_sun_hours.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.points is not None:
            self.points.to_hdf(
                self.results_directory / "points.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.daylight_factor is not None:
            self.daylight_factor.to_hdf(
                self.results_directory / "daylight_factor.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.annual_daylight_direct is not None:
            self.annual_daylight_direct.to_hdf(
                self.results_directory / "annual_daylight_direct.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.annual_daylight_total is not None:
            self.annual_daylight_total.to_hdf(
                self.results_directory / "annual_daylight_total.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.annual_irradiance_direct is not None:
            self.annual_irradiance_direct.to_hdf(
                self.results_directory / "annual_irradiance_direct.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.annual_irradiance_total is not None:
            self.annual_irradiance_total.to_hdf(
                self.results_directory / "annual_irradiance_total.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.daylight_autonomy is not None:
            self.daylight_autonomy.to_hdf(
                self.results_directory / "daylight_autonomy.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.continuous_daylight_autonomy is not None:
            self.continuous_daylight_autonomy.to_hdf(
                self.results_directory / "continuous_daylight_autonomy.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )
        if self.useful_daylight_illuminance_lower is not None:
            self.useful_daylight_illuminance_lower.to_hdf(
                self.results_directory / "useful_daylight_illuminance_lower.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )

        if self.useful_daylight_illuminance is not None:
            self.useful_daylight_illuminance.to_hdf(
                self.results_directory / "useful_daylight_illuminance.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )

        if self.useful_daylight_illuminance_upper is not None:
            self.useful_daylight_illuminance_upper.to_hdf(
                self.results_directory / "useful_daylight_illuminance_upper.hdf",
                "df",
                complevel=9,
                complib="blosc",
            )

        return None


def _load_pts(pts_file: Path) -> pd.DataFrame:
    """Return a DataFrame with point index along rows, and point coordinates and vectors along rows."""

    pts_file = pts_file if isinstance(pts_file, Path) else Path(pts_file)

    df = pd.read_csv(
        pts_file, header=None, names=["x", "y", "z", "vx", "vy", "vz"], sep="\s+"
    )
    df.columns = pd.MultiIndex.from_product([[pts_file.stem], df.columns])
    return df


def _load_res(res_file: Path) -> pd.Series:
    """Return a DataFrame with point index along rows, and illuminance value."""

    res_file = res_file if isinstance(res_file, Path) else Path(res_file)

    series = pd.read_csv(res_file, header=None, sep="\s+").squeeze()
    series.name = res_file.stem
    return series


def _load_ill(ill_file: Path, sun_up_hours_file: Path = None) -> pd.DataFrame:
    """Return a DataFrame with point index along columns, and hourly illuminance values along rows."""

    ill_file = ill_file if isinstance(ill_file, Path) else Path(ill_file)
    if not sun_up_hours_file:
        try:
            sun_up_hours_file = ill_file.parent / "sun-up-hours.txt"
        except FileNotFoundError as e:
            FileNotFoundError(
                "Cannot find sun-up-hours.txt file in the same directory as ill_file. Please add the directory to the inputs"
            )
    sun_up_hours_file = (
        sun_up_hours_file
        if isinstance(sun_up_hours_file, Path)
        else Path(sun_up_hours_file)
    )

    sun_up_hours = np.floor(
        pd.read_csv(sun_up_hours_file, header=None, index_col=None).squeeze().values
    ).astype(int)
    df = pd.read_csv(ill_file, sep="\s+", header=None, index_col=None).T
    df.index = sun_up_hours
    try:
        df = df.reindex(range(8760)).fillna(0).clip(lower=0)
    except ValueError as e:
        raise ValueError(
            f"The simulation results for {ill_file} are not shaped the same way as the sun-up-hours in {sun_up_hours_file}. If you have 4k results, you should have 4k sun-up-hours."
        )
    df.columns = pd.MultiIndex.from_product([[ill_file.stem], df.columns])

    return df


def _load_sql(sql_file: Path) -> pd.DataFrame:
    """Return a DataFrame with hourly values along rows and variables along columns."""

    sql_file = sql_file if isinstance(sql_file, Path) else Path(sql_file)

    sql_obj = SQLiteResult(sql_file.as_posix())

    def _flatten(container):
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in _flatten(i):
                    yield j
            else:
                yield i

    collections = list(
        _flatten(
            [
                sql_obj.data_collections_by_output_name(i)
                for i in sql_obj.available_outputs
            ]
        )
    )

    serieses = []
    headers = []
    for collection in collections:
        serieses.append(to_series(collection))
        variable = collection.header.metadata["type"]
        unit = collection.header.unit

        if "Surface" in collection.header.metadata.keys():
            element = "Surface"
            subelement = collection.header.metadata["Surface"]
        elif "System" in collection.header.metadata.keys():
            element = "System"
            subelement = collection.header.metadata["System"]
        elif "Zone" in collection.header.metadata.keys():
            element = "Zone"
            subelement = collection.header.metadata["Zone"]

        headers.append((element, subelement, f"{variable} ({unit})"))
    df = pd.concat(serieses, axis=1)
    df.columns = pd.MultiIndex.from_tuples(headers)
    df.sort_index(axis=1, inplace=True)
    return df
