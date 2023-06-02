from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.shelter import Point3D, Shelter
from ladybugtools_toolkit.ladybug_extension.epw import sun_position_list

from .. import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)
GENERIC_SHELTER = Shelter(
    [
        Point3D(-10, -10, 5),
        Point3D(-10, 10, 5),
        Point3D(10, 10, 5),
        Point3D(10, -10, 5),
    ]
)
SUNS = sun_position_list(EPW_OBJ)


def test_to_dict():
    """Test whether an object can be converted to a dictionary."""
    obj = GENERIC_SHELTER
    obj_dict = obj.to_dict()
    assert "_t" in obj_dict.keys()


def test_to_json():
    """Test whether an object can be converted to a json string."""
    obj = GENERIC_SHELTER
    obj_json = obj.to_json()
    assert '"_t":' in obj_json


def test_from_dict_native():
    """Test whether an object can be converted from a dictionary directly."""
    obj = GENERIC_SHELTER
    obj_dict = obj.to_dict()
    assert isinstance(Shelter.from_dict(obj_dict), Shelter)


def test_from_json_native():
    """Test whether an object can be converted from a json string directly."""
    obj = GENERIC_SHELTER
    obj_dict = obj.to_json()
    assert isinstance(Shelter.from_json(obj_dict), Shelter)


def test_set_porosity():
    """Test whether porosity setting works."""
    assert GENERIC_SHELTER.WindPorosity == 0
    assert GENERIC_SHELTER.RadiationPorosity == 0
    assert GENERIC_SHELTER.set_porosity(0.5).WindPorosity == 0.5
    assert GENERIC_SHELTER.set_porosity(0.5).RadiationPorosity == 0.5


def test_sky_exposure():
    """Test amount of sky exposure."""
    assert GENERIC_SHELTER.sky_exposure() == 0.31195840554592724
    assert GENERIC_SHELTER.set_porosity(0.5).sky_exposure() == 0.6559792027729636


def test_sun_exposure():
    """Test sun exposure"""
    assert sum(GENERIC_SHELTER.annual_sun_exposure(EPW_OBJ)) == 2041
    assert sum(GENERIC_SHELTER.set_porosity(0.5).annual_sun_exposure(EPW_OBJ)) == 3253


def test_wind_adjustment():
    """Test wind speed adjustment"""
    assert (
        sum(GENERIC_SHELTER.annual_effective_wind_speed(EPW_OBJ)) == 12495.516000000243
    )
    assert (
        sum(GENERIC_SHELTER.set_porosity(0.5).annual_effective_wind_speed(EPW_OBJ))
        == 21015.186000001042
    )
