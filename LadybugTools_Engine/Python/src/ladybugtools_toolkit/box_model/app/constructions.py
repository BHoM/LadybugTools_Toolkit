# Honeybee energy imports
import re

from honeybee_energy.lib.constructionsets import construction_set_by_identifier
from ladybug_epw import EPW


# Create construction set from EPW
def create_construction_set(epw, ashrae_vintage, constr_type):

  # Get climate zone from EPW 
  climate_zone = epw.climate_zone

  # Clean zone to be only numbers
  clean_zone = re.sub('[^0-9]', '', climate_zone)

  # Create ID
  cs_id = '{}_{}_{}'.format(clean_zone, ashrae_vintage, constr_type)

  # Get construction set
  construction_set = construction_set_by_identifier(cs_id)

  return construction_set