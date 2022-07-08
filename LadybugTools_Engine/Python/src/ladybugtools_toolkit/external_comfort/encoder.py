import json
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee._base import _Base as HB_Base
from honeybee_energy.material._base import _EnergyMaterialBase
from ladybug._datacollectionbase import BaseCollection
from ladybug.epw import EPW


class Encoder(json.JSONEncoder):
    def default(self, obj):

        # Path encoding
        if isinstance(obj, Path):
            return obj.as_posix()

        # NumPy encoding
        if isinstance(obj, (np.number, np.inexact, np.floating, np.complexfloating)):
            return float(obj)
        if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(obj)
        if isinstance(obj, (np.character)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Pandas encoding
        if isinstance(obj, (pd.DataFrame, pd.Series, pd.DatetimeIndex)):
            return obj.to_dict()

        # Ladybug encoding
        if isinstance(obj, (EPW, BaseCollection)):
            return obj.to_dict()

        # Honeybee encoding
        if isinstance(obj, HB_Base):
            return obj.to_dict()
        if isinstance(obj, _EnergyMaterialBase):
            return obj.to_dict()
        return super(Encoder, self).default(obj)
