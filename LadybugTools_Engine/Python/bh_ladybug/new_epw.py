from ladybug.epw import EPW
from .epw import BH_EPW
from .datacollection import BH_HourlyContinuousCollection, BH_MonthlyCollection

#######################
#  EXTENSION METHODS  #
#######################

# def print_yo(self: EPW) -> None:
#     print("yo")

def dry_bulb_temperature(self: EPW) -> BH_HourlyContinuousCollection:
    """
    Returns a BH_HourlyContinuousCollection of dry bulb temperature values.
    """
    _ = self._get_data_by_field(6)
    return BH_HourlyContinuousCollection(_.header, _.values)

# EPW.dry_bulb_temperature = dry_bulb_temperature
# EPW.print_yo = print_yo

setattr(EPW, 'dry_bulb_temperature', property(dry_bulb_temperature))

# EPW.__class__ = BH_EPW
