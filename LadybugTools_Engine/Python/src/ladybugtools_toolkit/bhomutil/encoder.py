import json
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee._base import _Base as HB_Base
from honeybee_energy.material._base import _EnergyMaterialBase
from ladybug._datacollectionbase import BaseCollection
from ladybug.epw import EPW


class BHoMEncoder(json.JSONEncoder):
    """A custom encoder for converting objects from this toolkit into serialisable JSON.

    Args:
        o (_type_):
            The object for conversion.
    """

    # pylint : disable=super-with-arguments;too-many-return-statements
    def default(self, o):
        # Path encoding
        if isinstance(o, Path):
            return o.as_posix()

        # NumPy encoding
        if isinstance(o, (np.number, np.inexact, np.floating, np.complexfloating)):
            return float(o)
        if isinstance(o, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(o)
        if isinstance(o, (np.character)):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        # Pandas encoding
        if isinstance(o, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
            return o.to_dict()

        # Ladybug encoding
        if isinstance(o, BaseCollection):
            return o.to_dict()
        if isinstance(o, EPW):
            return o.to_dict()

        # Honeybee encoding
        if isinstance(o, HB_Base):
            return o.to_dict()
        if isinstance(o, _EnergyMaterialBase):
            return o.to_dict()

        # Catch-all for any other object that has a "to_dict" method
        try:
            return o.to_dict()
        except AttributeError:
            try:
                return str(o)
            except Exception:  # pylint: disable=broad-except
                pass

        return super(BHoMEncoder, self).default(o)

    # pylint : enable=super-with-arguments;too-many-return-statements
