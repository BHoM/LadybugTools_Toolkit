from ladybugtools_toolkit.plot.colormap_sequential import colormap_sequential
from matplotlib.colors import BoundaryNorm, ListedColormap

UTCI_COLORMAP = ListedColormap(
    [
        "#262972",
        "#3452A4",
        "#3C65AF",
        "#37BCED",
        "#2EB349",
        "#F38322",
        "#C31F25",
        "#7F1416",
    ]
)
UTCI_COLORMAP.set_under("#0D104B")
UTCI_COLORMAP.set_over("#580002")
UTCI_LEVELS = [-40, -27, -13, 0, 9, 26, 32, 38, 46]
UTCI_LEVELS_IP = [-40, -17, 8, 32, 48, 83, 90, 100, 115]
UTCI_LABELS = [
    "Extreme Cold Stress",
    "Very Strong Cold Stress",
    "Strong Cold Stress",
    "Moderate Cold Stress",
    "Slight Cold Stress",
    "No Thermal Stress",
    "Moderate Heat Stress",
    "Strong Heat Stress",
    "Very Strong Heat Stress",
    "Extreme Heat Stress",
]
UTCI_BOUNDARYNORM = BoundaryNorm(UTCI_LEVELS, UTCI_COLORMAP.N)
UTCI_BOUNDARYNORM_IP = BoundaryNorm(UTCI_LEVELS_IP, UTCI_COLORMAP.N)

DBT_COLORMAP = colormap_sequential("white", "#bc204b")
RH_COLORMAP = colormap_sequential("white", "#8db9ca")
MRT_COLORMAP = colormap_sequential("white", "#6d104e")
WS_COLORMAP = colormap_sequential(
    "#d0e8e4",
    "#8db9ca",
    "#006da8",
    "#24135f",
)
