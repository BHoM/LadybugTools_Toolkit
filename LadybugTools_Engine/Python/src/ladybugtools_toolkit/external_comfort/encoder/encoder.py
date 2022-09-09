import json
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee._base import _Base as HB_Base
from honeybee_energy.material._base import _EnergyMaterialBase
from ladybug._datacollectionbase import BaseCollection
from ladybug.epw import EPW


class Encoder(json.JSONEncoder):
    """A custom encoder for converting LadybugTools_Toolkit.external_comfort objects into
        serialisable JSON.

    Args:
        o (_type_):
            The object for conversion.
    """

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
        if isinstance(o, (EPW, BaseCollection)):
            return o.to_dict()

        # Honeybee encoding
        if isinstance(o, HB_Base):
            return o.to_dict()
        if isinstance(o, _EnergyMaterialBase):
            return o.to_dict()

        # Catch-all fr any object that has a "to_dict" method
        try:
            return o.to_dict()
        except AttributeError:
            pass

        return super(Encoder, self).default(o)
