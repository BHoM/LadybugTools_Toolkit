import pytest
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.epw import EPW
from ladybug_comfort.collection.utci import UTCI
from ladybugtools_toolkit.external_comfort.utci import summarise_utci
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


def test_summarise_utci_collection():
    """_"""
    assert (
        summarise_utci(
            LB_UTCI_COLLECTION, AnalysisPeriod(st_month=6, end_month=3), sep=" "
        )
        == 'In this summary, thermal comfort or periods experiencing no thermal stress, are UTCI values of between 9°C and 26°C. For Jun 01 to Mar 31 between 00:00 and 23:59, "No thermal stress" is expected for 2633 out of a possible 7296 hours (36.1%). "Cold stress" is expected for 4655 hours (63.8%). "Heat stress" is expected for 8 hours (0.1%).'
    )
