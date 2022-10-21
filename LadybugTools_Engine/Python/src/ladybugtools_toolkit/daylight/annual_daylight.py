from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ladybugtools_toolkit.honeybee_extension.results.load_ill import load_ill
from ladybugtools_toolkit.honeybee_extension.results.load_pts import load_pts
from ladybugtools_toolkit.honeybee_extension.results.make_annual import \
    make_annual


@dataclass(init=True, repr=True)
class AnnualDaylight:
    # define properties of this object
    directory: Path = field(init=True)
    ill: pd.DataFrame = field(init=False)
    pts: pd.DataFrame = field(init=False)
    sun_up_hours: List[float] = field(init=False)
    rooms: List[str] = field(init=False)

    def __post_init__(self):
        # after object created, populate the properties
        self.directory = Path(self.directory)
        self.pts = self._pts()
        self.ill = self._ill()
        self.sun_up_hours = self._sun_up_hours()
        self.rooms = np.unique(self.ill.columns.get_level_values(0))

    @staticmethod
    def find_files(directory: Path, extension: str) -> List[Path]:
        """Find all files of type in a given directory.

        Args:
            directory (Path):
                A directory as a Path object.
            extension (str):
                The extension of the file type to search for.

        Returns:
            List[Path]:
                A list iof matching files.
        """
        if not extension.startswith("."):
            extension = "." + extension
        return list(Path(directory).glob(f"**/*{extension}"))

    def _pts_files(self) -> List[Path]:
        """Find all *.pts files in the current object.

        Returns:
            List[Path]:
                A list of pts files.
        """        
        return self.find_files(self.directory / "model" / "grid", ".pts")

    def _pts(self) -> pd.DataFrame:
        """Get the points simulated for this daylight object.

        Returns:
            pd.DataFrame:
                A dataframe containing point locations.
        """
        return load_pts(self._pts_files())

    def _ill_files(self) -> List[Path]:
        """Get the points simulated for this daylight object.

        Returns:
            List[Path]:
                A list of ill files.
        """
        return self.find_files(self.directory / "results", ".ill")

    def _ill(self) -> pd.DataFrame:
        """Get the illuminance simulated for this daylight object.

        Returns:
            pd.DataFrame:
                A dataframe containing illuminance values.
        """
        return make_annual(load_ill(self._ill_files())).fillna(0)

    def _sun_up_hours_file(self) -> Path:
        """Get the sun-up-hours file for this daylight object.

        Returns:
            Path:
                A sun-up-hours file.
        """
        return self.find_files(self.directory / "results", ".txt")[0]

    def _sun_up_hours(self) -> List[float]:
        """Get a list of hours of the year where sun is up.

        Returns:
            List[float]:
                A list of sun-up hours.
        """
        with open(self._sun_up_hours_file(), "r", encoding="utf-8") as fp:
            return [float(i.strip()) for i in fp.readlines()]

    def lbt_daylight_autonomy(self) -> List[Path]:
        """Get the daylight autonomy metrics provided by LadybugTools.

        Returns:
            Dict[str, List[float]]:
                A dictionary containing <space_name>: [<daylight_autonomy_values>].
        """
        da_files = self.find_files(self.directory / "metrics" / "da", ".da")
        d = {}
        for da_file in da_files:
            with open(da_file, "r", encoding="utf-8") as fp:
                d[da_file.stem] = [float(i.strip()) for i in fp.readlines()]
        return d

    def daylight_autonomy(self, occupied_hours: Tuple[int] = (9, 10, 11, 12, 13, 14, 15, 16, 17)) -> Dict[str, List[float]]:
        """Calculate daylight autonomy.

        Args:
            occupied_hours (Tuple[int], optional):
                A list of hours occupied (when to asses for daylight autonomy). Defaults to (9, 10, 11, 12, 13, 14, 15, 16, 17).

        Returns:
            Dict[str, List[float]]:
                A dictionary containing <space_name>: [<daylight_autonomy_values>].
        """        
        occ = self.ill.index.hour.isin(occupied_hours)
        
        d = {}
        for room in self.rooms:
            _ill = self.ill.loc[occ][room]
            d[room] = ((_ill > 300).sum(axis=0) / len(_ill)).tolist()
        return d
    


if __name__ == "__main__":

    ad = AnnualDaylight(Path(r"C:\Users\tgerrish\simulation\unnamed\annual_daylight"))

    print(ad.daylight_autonomy())
