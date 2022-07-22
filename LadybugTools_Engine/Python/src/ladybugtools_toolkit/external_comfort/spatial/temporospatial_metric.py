from enum import Enum, auto


class TemporospatialMetric(Enum):
    """A list of spatial metrics describing a spatial case."""

    DBT = auto()
    EVAP_CLG = auto()
    MRT = auto()
    RAD_DIFFUSE = auto()
    RAD_DIRECT = auto()
    RAD_TOTAL = auto()
    RH = auto()
    UTCI_CALCULATED = auto()
    UTCI_INTERPOLATED = auto()
    WD = auto()
    WS = auto()
