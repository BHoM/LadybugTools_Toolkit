"""Module for creating shelter objects."""

# pylint: disable=E0401
import json
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from ._shelterbase import Shelter

# pylint: enable=E0401


class TreeShelter(Enum):
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

        idx = pd.date_range("2017-01-01", periods=8760, freq="60min")
        if tree_config["deciduous"]:
            vals = np.stack(
                [
                    np.where(
                        (idx < pd.to_datetime(
                            tree_config["regrowth_period"][0])) | (
                            idx > pd.to_datetime(
                                tree_config["drop_period"][1])),
                        tree_config["porosity_bare"],
                        np.nan,
                    ),
                    np.where(
                        (idx > pd.to_datetime(
                            tree_config["regrowth_period"][1])) & (
                            idx < pd.to_datetime(
                                tree_config["drop_period"][0])),
                        tree_config["porosity_leafed"],
                        np.nan,
                    ),
                ])
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
