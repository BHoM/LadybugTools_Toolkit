"""Module for creating shelter objects."""
# pylint: disable=E0401
import json
import warnings
from enum import Enum
from pathlib import Path

# pylint: enable=E0401

import numpy as np
import pandas as pd
from ..bhom import decorator_factory
from ._shelterbase import Shelter


@decorator_factory()
def north_south_linear() -> Shelter:
    """Predefined north-south linear shelter."""
    return Shelter.from_overhead_linear(
        width=3, height_above_ground=3.5, length=2000, angle=0
    )


@decorator_factory()
def east_west_linear() -> Shelter:
    """Predefined east-west linear shelter."""
    return Shelter.from_overhead_linear(
        width=3, height_above_ground=3.5, length=2000, angle=90
    )


@decorator_factory()
def northeast_southwest_linear() -> Shelter:
    """Predefined northeast-southwest linear shelter."""
    return Shelter.from_overhead_linear(
        width=3, height_above_ground=3.5, length=2000, angle=45
    )


@decorator_factory()
def northwest_southeast_linear() -> Shelter:
    """Predefined northwest-southeast linear shelter."""
    return Shelter.from_overhead_linear(
        width=3, height_above_ground=3.5, length=2000, angle=135
    )


@decorator_factory()
def overhead_large() -> Shelter:
    return Shelter.from_overhead_circle(radius=5, height_above_ground=3.5)


@decorator_factory()
def overhead_small() -> Shelter:
    return Shelter.from_overhead_circle(radius=1.5, height_above_ground=3.5)


@decorator_factory()
def north_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=0,
    )


@decorator_factory()
def northeast_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=45,
    )


@decorator_factory()
def east_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=90,
    )


@decorator_factory()
def southeast_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=135,
    )


@decorator_factory()
def south_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=180,
    )


@decorator_factory()
def southwest_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=225,
    )


@decorator_factory()
def west_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=270,
    )


@decorator_factory()
def northwest_wall() -> Shelter:
    return Shelter.from_adjacent_wall(
        distance_from_wall=1,
        wall_height=2,
        wall_length=2,
        angle=315,
    )


class TreeSpecies(Enum):
    """_"""

    ACER_PLATANOIDES = "Norway maple"
    AESCULUS_HIPPOCASTANUM = "European horse-chestnut"
    BETULA_PENDULA = "Silver birch"
    CARPINUS_BETULUS = "European hornbeam"
    FAGUS_SYLVATICA = "European beech"
    FRAXINUS_AMERICANA = "White ash"
    MAGNOLIA_KOBUS = "Kobushi magnolia"
    MALUS_FLORIBUNDA = "Japanese flowering crabapple"
    PHOENIX_DACTYLIFERA = "Date palm"
    PRUNUS_CERASIFERA = "Cherry plum"
    PRUNUS_SERRULATA = "East Asian Cherry"
    QUERCUS_PALUSTRIS = "Spanish oak"
    TILIA_X_EUCHLORA = "Caucasian lime"

    @decorator_factory()
    def shelter(self, northern_hemisphere: bool = True) -> Shelter:
        """Get a shelter object for this tree species.

        Args:
            northern_hemisphere (bool, optional):
                If True, then the tree will have leafcover over the northern
                hemisphere summer months. If False then this will be shifted by
                6-months to approximate the change for the southern hemisphere.
                Defaults to True.

        Returns:
            Shelter:
                A Shelter object.
        """

        # load json file
        with open(
            Path(__file__).parent.parent.parent / "data" / r"vegetation.json", "r"
        ) as fp:
            tree_config = json.load(fp)[self.name]

        idx = pd.date_range("2017-01-01", periods=8760, freq="60T")
        if tree_config["deciduous"]:
            vals = np.stack(
                [
                    np.where(
                        (idx < pd.to_datetime(tree_config["regrowth_period"][0]))
                        | (idx > pd.to_datetime(tree_config["drop_period"][1])),
                        tree_config["porosity_bare"],
                        np.nan,
                    ),
                    np.where(
                        (idx > pd.to_datetime(tree_config["regrowth_period"][1]))
                        & (idx < pd.to_datetime(tree_config["drop_period"][0])),
                        tree_config["porosity_leafed"],
                        np.nan,
                    ),
                ]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vals = np.nanmax(vals, axis=0)
            _porosity = (
                pd.Series(vals, index=idx)
                .interpolate(method="polynomial", order=2)
                .values
            )
        else:
            _porosity = np.ones(8760) * tree_config["porosity_leafed"]

        if not northern_hemisphere:
            _porosity = np.roll(_porosity, int(8760 / 2))

        return Shelter.from_overhead_circle(
            radius=tree_config["average_radius"],
            height_above_ground=tree_config["average_height"],
        ).set_porosity(_porosity.tolist())
