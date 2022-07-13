from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import Temperature
from ladybug.epw import EPW
from ladybug_comfort.collection.solarcal import HorizontalSolarCal
from ladybug_comfort.parameter.solarcal import SolarCalParameter

from ..ladybug_extension.datacollection import from_series, to_series
from .encoder import Encoder
from .external_comfort import ExternalComfort
from .simulate import solar_radiation, surface_temperature


@dataclass(frozen=True)
class ExternalComfortResult:
    external_comfort: ExternalComfort = field(init=True, repr=True, compare=True)

    shaded_below_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_above_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_direct_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_diffuse_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    shaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )

    unshaded_below_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_above_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_direct_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_diffuse_radiation: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_longwave_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )
    unshaded_mean_radiant_temperature: HourlyContinuousCollection = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> ExternalComfortResult:
        """Calculate the mean radiant tempertaure, and constituent parts of this value from the External Comfort configuration."""

        print(f"- Running external comfort calculation for {self}")

        # Run EnergyPlus and Radiance simulations
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            for f in [solar_radiation, surface_temperature]:
                results.append(
                    executor.submit(
                        f, self.external_comfort.model, self.external_comfort.epw
                    )
                )

        # Populate simulation results
        for x in results:
            for k, v in x.result().items():
                object.__setattr__(self, k, v)

        # Populate calculated results
        object.__setattr__(
            self,
            "shaded_longwave_mean_radiant_temperature",
            self._radiant_temperature_from_collections(
                [
                    self.shaded_below_temperature,
                    self.shaded_above_temperature,
                ],
                [0.5, 0.5],
            ),
        )

        object.__setattr__(
            self,
            "unshaded_longwave_mean_radiant_temperature",
            self._radiant_temperature_from_collections(
                [
                    self.unshaded_below_temperature,
                    self.unshaded_above_temperature,
                ],
                [0.5, 0.5],
            ),
        )

        object.__setattr__(
            self,
            "shaded_mean_radiant_temperature",
            self._mean_radiant_temperature(
                self.external_comfort.epw,
                self.shaded_longwave_mean_radiant_temperature,
                self.shaded_direct_radiation,
                self.shaded_diffuse_radiation,
            ),
        )

        object.__setattr__(
            self,
            "unshaded_mean_radiant_temperature",
            self._mean_radiant_temperature(
                self.external_comfort.epw,
                self.unshaded_longwave_mean_radiant_temperature,
                self.unshaded_direct_radiation,
                self.unshaded_diffuse_radiation,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this object as a dictionary

        Returns:
            Dict: The dict representation of this object.
        """

        attributes = [
            "external_comfort",
            "shaded_below_temperature",
            "shaded_above_temperature",
            "shaded_direct_radiation",
            "shaded_diffuse_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "unshaded_below_temperature",
            "unshaded_above_temperature",
            "unshaded_direct_radiation",
            "unshaded_diffuse_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
        ]
        return {attribute: getattr(self, attribute) for attribute in attributes}

    def to_json(self, file_path: str) -> Path:
        """Write the content of this object to a JSON file

        Returns:
            Path: The path to the newly created JSON file.
        """

        file_path: Path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True, parents=True)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, cls=Encoder, indent=4)

        return file_path

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.external_comfort.identifier})"

    def to_dataframe(self) -> pd.DataFrame:
        """Create a dataframe from the simulation results.

        Returns:
            pd.DataFrame: A dataframe containing the simulation results.
        """

        attributes = [
            "shaded_below_temperature",
            "shaded_above_temperature",
            "shaded_direct_radiation",
            "shaded_diffuse_radiation",
            "shaded_longwave_mean_radiant_temperature",
            "shaded_mean_radiant_temperature",
            "unshaded_below_temperature",
            "unshaded_above_temperature",
            "unshaded_direct_radiation",
            "unshaded_diffuse_radiation",
            "unshaded_longwave_mean_radiant_temperature",
            "unshaded_mean_radiant_temperature",
        ]
        series: List[pd.Series] = []
        for attribute in attributes:
            series.append(to_series(getattr(self, attribute)))
        return pd.concat(
            series,
            axis=1,
            keys=[f"{self.__class__.__name__} - {i}" for i in attributes],
        )
