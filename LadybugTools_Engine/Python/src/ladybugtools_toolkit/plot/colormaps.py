"""Default colormaps"""

from .utilities import colormap_sequential

DBT_COLORMAP = colormap_sequential("white", "#bc204b")
RH_COLORMAP = colormap_sequential("white", "#8db9ca")
MRT_COLORMAP = colormap_sequential("white", "#6d104e")
WS_COLORMAP = colormap_sequential(
    "#d0e8e4",
    "#8db9ca",
    "#006da8",
    "#24135f",
)
UTCI_DIFFERENCE_COLORMAP = colormap_sequential("#00A9E0", "w", "#702F8A")
UTCI_DISTANCE_TO_COMFORTABLE = colormap_sequential("#00A9E0", "w", "#ba000d")
_ = colormap_sequential(
    "#E40303", "#FF8C00", "#FFED00", "#008026", "#24408E", "#732982"
)
