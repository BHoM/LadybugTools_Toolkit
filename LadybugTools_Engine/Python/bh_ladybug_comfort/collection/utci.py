from __future__ import annotations

from ladybug._datacollectionbase import BaseCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybug_comfort.collection.utci import UTCI
from ladybug_comfort.parameter.utci import UTCIParameter

from ...bh_ladybug.datacollectionimmutable import BH_HourlyContinuousCollectionImmutable


class BH_UTCI(UTCI):
    def __init__(
        self,
        air_temperature: BaseCollection,
        rel_humidity: BaseCollection,
        rad_temperature: BaseCollection = None,
        wind_speed: BaseCollection = None,
        comfort_parameter: UTCIParameter = None,
    ) -> BH_UTCI:
        super().__init__(
            air_temperature,
            rel_humidity,
            rad_temperature,
            wind_speed,
            comfort_parameter,
        )
        _model = super()._model
        __slots__ = super().__slots__

    @property
    def universal_thermal_climate_index(self) -> BH_HourlyContinuousCollectionImmutable:
        """A Data Collection of Universal Thermal Climate Index (UTCI) in C."""
        base_collection = super().universal_thermal_climate_index
        base_collection.__class__ = BH_HourlyContinuousCollectionImmutable
        return base_collection