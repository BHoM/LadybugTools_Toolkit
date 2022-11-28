import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List

import numpy as np
from honeybee.face import Face3D
from honeybee_radiance.modifier.material.glass import Glass
from honeybee_radiance.modifier.material.glass import Material as HBR_Material
from ladybugtools_toolkit.ladybug_extension.analysis_period import (
    AnalysisPeriod,
    to_datetimes,
)

from ..bhomutil.bhom_object import BHoMObject


def modify_vegetation_transmissivity_by_season(
    modifier: HBR_Material,
    analysis_period: AnalysisPeriod,
    monthly_transmissivity: Dict[int, float],
    inplace: bool = False,
) -> HBR_Material:
    """Modify a Radiance modifier to adjust porosity based on the analysis period provided.

    Args:
        modifier (HBR_Material):
            A Honeybee Raidance modifier material.
        analysis_period (AnalysisPeriod):
            A Ladybug AnalysisPeriod.
        monthly_transmissivity (Dict[int, float]):
            A dictionary containing a lookup for monthly transmissivity values.
            An example is given below.
        inplace (bool, optional):
            If True, modify the material where it currently exists instead of
            creating and returning a new material. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        HBR_Material: _description_
    """
    if not isinstance(modifier, Glass):
        raise ValueError(
            f"The modifier that is being modified is not of type ({Glass})."
        )

    if list(monthly_transmissivity.keys()) != range(1, 13, 1):
        raise ValueError(
            "The monthly_transmissivity dictionary does not contain values for each month of the year."
        )

    datetimes = to_datetimes(analysis_period)
    unique_months = np.unique([i.month for i in datetimes])

    # get the average transmissivity for the months in the given analysis period
    avg_transmissivity = np.mean([monthly_transmissivity[i] for i in unique_months])

    if inplace:
        # unlock the provided material
        modifier.unlock()
        modifier.r_transmissivity = avg_transmissivity
        modifier.g_transmissivity = avg_transmissivity
        modifier.b_transmissivity = avg_transmissivity
        modifier.lock()
        return None
    new_modifier = deepcopy(modifier)
    new_modifier.unlock()
    new_modifier.r_transmissivity = avg_transmissivity
    new_modifier.g_transmissivity = avg_transmissivity
    new_modifier.b_transmissivity = avg_transmissivity
    new_modifier.lock()
    return new_modifier


class VegetationCategory(Enum):
    PARK_TREE = auto()
    STREET_TREE = auto()
    PALM_TREE = auto()
    FRUIT_TREE = auto()
    WATERFRONT_TREE = auto()


class VegetationShape(Enum):
    HEMISPHERICAL = auto()
    ELLIPSOID = auto()
    CONICAL_TRUNK = auto()
    ELLIPSOID_TRUNK = auto()
    HALF_ELLIPSOID_TRUNK = auto()


@dataclass(init=True, repr=True, eq=True)
class Vegetation(BHoMObject):
    """An object describing a piece of vegetation."""

    common_name: str = field(init=True, repr=True, compare=True)
    category: VegetationCategory = field(init=True, repr=False, compare=True)
    fully_grown_height: float = field(init=True, repr=False, compare=True)
    trunk_height: float = field(init=True, repr=False, compare=True)
    canopy_height: float = field(init=True, repr=False, compare=True)
    canopy_diameter: float = field(init=True, repr=False, compare=True)
    canopy_widest_point: float = field(
        init=True, repr=False, compare=True
    )  # abstract value denoting height above start of canopy whee widest point is reached
    deciduous: bool = field(init=True, repr=True, compare=True)
    porosity: float = field(init=True, repr=False, compare=True)
    latin_name: str = field(init=True, repr=False, compare=False, default="")
    trunk_radius: float = field(init=False, repr=False, compare=True, default=0.2)

    _t: str = field(
        init=False,
        repr=False,
        compare=True,
        default="BH.oM.LadybugTools.Vegetation",
    )

    def __post_init__(self):
        # wrap methods within this class
        1 + 1  # pylint: disable=pointless-statement
        super().__post_init__()  # pylint: disable=useless-parent-delegation

    @property
    def shape(self) -> VegetationShape:
        """Estimate the vegetation shape"""
        if self.trunk_height <= 0.5:
            if np.isclose(self.canopy_height, self.canopy_diameter, rtol=2):
                return VegetationShape.ELLIPSOID
            if self.canopy_diameter > self.canopy_height:
                return VegetationShape.HEMISPHERICAL
        if self.canopy_widest_point >= 0.75:
            return VegetationShape.CONICAL_TRUNK
        if np.isclose(self.canopy_height, self.canopy_diameter, rtol=3):
            return VegetationShape.ELLIPSOID_TRUNK
        return VegetationShape.HALF_ELLIPSOID_TRUNK

    def honeybee_faces(self) -> List[Face3D]:
        """Create the vegetation object as a set of Honeybee faces."""

        # TODO - Make functional method for creating honeybee geometry from the vegetation objects.
        warnings.warn(
            "This method doesnt currently return a set of faces as LB Geometry can't handle ellipsoids and nsplitting. An alternative needs to be found!"
        )
        return None


class Vegetations(Enum):
    """
    A set of vegetation types, and their associated properties.
    """

    ARABIAN_GUM_TREE = Vegetation(
        common_name="Arabian gum tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=14.0,
        trunk_height=4.1,
        canopy_height=9.9,
        canopy_diameter=13.5,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Acacia arabica",
    )
    UMBRELLA_THORN = Vegetation(
        common_name="Umbrella thorn",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=6.0,
        trunk_height=3.5,
        canopy_height=2.5,
        canopy_diameter=7.1,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Acacia tortilis",
    )
    WHITE_SAXAUL = Vegetation(
        common_name="White saxaul",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=4.8,
        trunk_height=0.0,
        canopy_height=4.8,
        canopy_diameter=3.8,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Haloxylon persicum",
    )
    WILD_COTTON_TREE = Vegetation(
        common_name="Wild cotton tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=7.0,
        trunk_height=0.0,
        canopy_height=7.0,
        canopy_diameter=6.4,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Hibiscus tiliaceus",
    )
    HENNA = Vegetation(
        common_name="Henna",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=3.5,
        trunk_height=0.9,
        canopy_height=2.6,
        canopy_diameter=1.0,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.2,
        latin_name="Lawsonisa inermis",
    )
    BEN_TREE = Vegetation(
        common_name="Ben tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=10.0,
        trunk_height=2.7,
        canopy_height=7.3,
        canopy_diameter=10.2,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Moringa peregrina",
    )
    OLIVE_TREE = Vegetation(
        common_name="Olive tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=6.8,
        trunk_height=2.5,
        canopy_height=4.2,
        canopy_diameter=7.9,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.5,
        latin_name="Olea europaea",
    )
    PALO_VERDE = Vegetation(
        common_name="Palo verde",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=5.0,
        trunk_height=1.5,
        canopy_height=3.5,
        canopy_diameter=8.2,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Parkinsonia aculeata",
    )
    YELLOW_FLAME_TREE = Vegetation(
        common_name="Yellow flame tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=12.5,
        trunk_height=5.4,
        canopy_height=7.1,
        canopy_diameter=18.4,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.5,
        latin_name="Peltophorum pterocarpum",
    )
    INGA_DULCE = Vegetation(
        common_name="Inga dulce",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=12.5,
        trunk_height=2.4,
        canopy_height=10.1,
        canopy_diameter=14.2,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Pithecellobium dulce",
    )
    GHAF = Vegetation(
        common_name="Ghaf",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=9.0,
        trunk_height=3.0,
        canopy_height=6.0,
        canopy_diameter=6.4,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.8,
        latin_name="Prosopis cineraria",
    )
    TOOTHBRUSH_TREE = Vegetation(
        common_name="Toothbrush tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=2.5,
        trunk_height=0.7,
        canopy_height=1.8,
        canopy_diameter=3.4,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Salvadora persica",
    )
    YELLOW_TABEBUIA = Vegetation(
        common_name="Yellow tabebuia",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=10.0,
        trunk_height=4.4,
        canopy_height=5.6,
        canopy_diameter=10.3,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Tabebuia argentea",
    )
    TAMARIX = Vegetation(
        common_name="Tamarix",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=9.0,
        trunk_height=0.0,
        canopy_height=9.0,
        canopy_diameter=17.3,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.5,
        latin_name="Tamarix aphylla",
    )
    ROHIDA = Vegetation(
        common_name="Rohida",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=10.0,
        trunk_height=5.2,
        canopy_height=4.8,
        canopy_diameter=12.5,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.5,
        latin_name="Tecomella undulata",
    )
    CHASTE_TREE = Vegetation(
        common_name="Chaste tree",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=5.0,
        trunk_height=0.7,
        canopy_height=4.3,
        canopy_diameter=4.1,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Vitex agnus-castus",
    )
    SIDR = Vegetation(
        common_name="Sidr",
        category=VegetationCategory.PARK_TREE,
        fully_grown_height=7.5,
        trunk_height=2.0,
        canopy_height=5.5,
        canopy_diameter=8.8,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.5,
        latin_name="Ziziphus spina-christi",
    )
    LEBBEK_TREE = Vegetation(
        common_name="Lebbek tree",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=12.0,
        trunk_height=4.0,
        canopy_height=8.0,
        canopy_diameter=9.7,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Albizia lebbeck",
    )
    NEEM = Vegetation(
        common_name="Neem",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=9.0,
        trunk_height=2.2,
        canopy_height=6.8,
        canopy_diameter=5.0,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Azadirachta indica",
    )
    SPINY_BUCIDA = Vegetation(
        common_name="Spiny bucida",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=6.0,
        trunk_height=1.6,
        canopy_height=4.4,
        canopy_diameter=3.7,
        canopy_widest_point=0.2,
        deciduous=False,
        porosity=0.5,
        latin_name="Bucida molinetii",
    )
    GEIGER_TREE = Vegetation(
        common_name="Geiger tree",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=10.0,
        trunk_height=3.4,
        canopy_height=6.6,
        canopy_diameter=6.3,
        canopy_widest_point=0.3,
        deciduous=False,
        porosity=0.2,
        latin_name="Cordia sebestena",
    )
    JASMINE_TREE = Vegetation(
        common_name="Jasmine tree",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=10.0,
        trunk_height=3.5,
        canopy_height=6.5,
        canopy_diameter=6.3,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.8,
        latin_name="Millingtonia hortensis",
    )
    INDIAN_ALMOND = Vegetation(
        common_name="Indian almond",
        category=VegetationCategory.STREET_TREE,
        fully_grown_height=14.0,
        trunk_height=4.6,
        canopy_height=9.4,
        canopy_diameter=13.7,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Terminalia catappa",
    )
    DATE_PALM = Vegetation(
        common_name="Date palm",
        category=VegetationCategory.PALM_TREE,
        fully_grown_height=20.0,
        trunk_height=7.9,
        canopy_height=12.1,
        canopy_diameter=10.4,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Phoenix dactylifera",
    )
    MEXICAN_FAN_PALM = Vegetation(
        common_name="Mexican fan palm",
        category=VegetationCategory.PALM_TREE,
        fully_grown_height=9.0,
        trunk_height=5.7,
        canopy_height=3.3,
        canopy_diameter=2.8,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.5,
        latin_name="Washingtonia robusta",
    )
    LEMON_TREE = Vegetation(
        common_name="Lemon tree",
        category=VegetationCategory.FRUIT_TREE,
        fully_grown_height=4.5,
        trunk_height=1.5,
        canopy_height=3.0,
        canopy_diameter=5.0,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.2,
        latin_name="Citrus ssp.",
    )
    MANGO_TREE = Vegetation(
        common_name="Mango tree",
        category=VegetationCategory.FRUIT_TREE,
        fully_grown_height=7.0,
        trunk_height=2.2,
        canopy_height=4.8,
        canopy_diameter=8.4,
        canopy_widest_point=0.5,
        deciduous=False,
        porosity=0.2,
        latin_name="Mangifera indica",
    )
    BANANA_TREE = Vegetation(
        common_name="Banana tree",
        category=VegetationCategory.FRUIT_TREE,
        fully_grown_height=2.5,
        trunk_height=0.7,
        canopy_height=1.8,
        canopy_diameter=1.3,
        canopy_widest_point=1,
        deciduous=False,
        porosity=0.5,
        latin_name="Musa x paradisiaca",
    )
    GUAVA_TREE = Vegetation(
        common_name="Guava tree",
        category=VegetationCategory.FRUIT_TREE,
        fully_grown_height=6.5,
        trunk_height=3.3,
        canopy_height=3.2,
        canopy_diameter=4.8,
        canopy_widest_point=0.2,
        deciduous=False,
        porosity=0.5,
        latin_name="Psidium guajava",
    )
    RED_DATE_TREE = Vegetation(
        common_name="Red date tree",
        category=VegetationCategory.FRUIT_TREE,
        fully_grown_height=7.0,
        trunk_height=3.4,
        canopy_height=3.6,
        canopy_diameter=2.4,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Ziziphus jujuba",
    )
    GREY_MANGROVE = Vegetation(
        common_name="Grey mangrove",
        category=VegetationCategory.WATERFRONT_TREE,
        fully_grown_height=2.5,
        trunk_height=0.7,
        canopy_height=1.8,
        canopy_diameter=4.2,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Avicennia marina",
    )
    RED_MANGROVE = Vegetation(
        common_name="Red mangrove",
        category=VegetationCategory.WATERFRONT_TREE,
        fully_grown_height=12.5,
        trunk_height=5.9,
        canopy_height=6.6,
        canopy_diameter=20.9,
        canopy_widest_point=0,
        deciduous=False,
        porosity=0.2,
        latin_name="Rhizophora mucronata",
    )
