from enum import Enum


class SpatialMetric(Enum):
    """A list of metrics that can be represented spatially."""

    RAD_DIFFUSE = "Diffuse radiation (W/m²)"
    RAD_DIRECT = "Direct radiation (W/m²)"
    RAD_TOTAL = "Total radiation (W/m²)"

    DBT_EPW = "Dry-bulb temperature (°C) - from EPW"
    RH_EPW = "Relative humidity (%) - from EPW"

    WD_EPW = "Wind direction (deg - from EPW"
    WS_EPW = "Wind speed (m/s) - from EPW"
    WS_CFD = "Wind speed (m/s) - from CFD"

    EVAP_CLG = "Evaporative cooling magnitude (0-1)"
    DBT_EVAP = "Dry-bulb temperature (°C) - inc. moisture effects"
    RH_EVAP = "Relative humidity (%) - inc. moisture effects"

    MRT_INTERPOLATED = "Mean radiant temperature (°C) - interpolated"

    UTCI_CALCULATED = "Universal thermal climate index (°C) - calculated"
    UTCI_INTERPOLATED = "Universal thermal climate index (°C) - interpolated"

    # single value frames here - used only for consistency in loading data
    SKY_VIEW = "Sky view (%)"
    POINTS = "Points (x, y, z)"
