from matplotlib.colors import BoundaryNorm, ListedColormap

UTCI_LOCAL_COLORMAP = ListedColormap(
    [
        "#262972",
        "#3452A4",
        "#3C65AF",
        "#37BCED",
        "#2EB349",
        "#D3D317",
        "#F38322",
        "#C31F25",
        "#7F1416",
    ]
)
UTCI_LOCAL_COLORMAP.set_under("#0D104B")
UTCI_LOCAL_COLORMAP.set_over("#580002")
UTCI_LOCAL_LEVELS = [-40, -27, -13, 0, 9, 26, 28, 32, 38, 46]
UTCI_LOCAL_LEVELS_IP = [-40, -17, 8, 32, 48, 79, 83, 90, 100, 115]
UTCI_LOCAL_LABELS = [
    "Extreme Cold Stress",
    "Very Strong Cold Stress",
    "Strong Cold Stress",
    "Moderate Cold Stress",
    "Slight Cold Stress",
    "No Thermal Stress",
    "Slight Heat Stress",
    "Moderate Heat Stress",
    "Strong Heat Stress",
    "Very Strong Heat Stress",
    "Extreme Heat Stress",
]
UTCI_LOCAL_BOUNDARYNORM = BoundaryNorm(UTCI_LOCAL_LEVELS, UTCI_LOCAL_COLORMAP.N)
UTCI_LOCAL_BOUNDARYNORM_IP = BoundaryNorm(UTCI_LOCAL_LEVELS_IP, UTCI_LOCAL_COLORMAP.N)