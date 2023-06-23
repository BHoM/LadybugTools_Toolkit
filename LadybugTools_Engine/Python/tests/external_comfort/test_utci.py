import pytest
from ladybug.epw import EPW
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.external_comfort.utci.calculate import utci

from .. import EPW_FILE

EPW_OBJ = EPW(EPW_FILE)

LB_UTCI_COLLECTION = UTCI(
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.relative_humidity,
    EPW_OBJ.dry_bulb_temperature,
    EPW_OBJ.wind_speed,
).universal_thermal_climate_index


def test_utci():
    """_"""
    assert utci(
        EPW_OBJ.dry_bulb_temperature.values,
        EPW_OBJ.relative_humidity.values,
        EPW_OBJ.dry_bulb_temperature.values,
        EPW_OBJ.wind_speed.values,
    ).mean() == pytest.approx(LB_UTCI_COLLECTION.average, rel=2)
