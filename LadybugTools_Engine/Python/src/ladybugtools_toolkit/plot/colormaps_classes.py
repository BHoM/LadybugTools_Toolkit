from ladybugtools_toolkit.plot.colormaps_class import UTCIColorScheme


class UTCIColorSchemes(UTCIColorScheme):
    """A list of pre-defined UTCI color scheme objects."""

    UTCI_Original = UTCIColorScheme(
        name = "Original",
        levels = [-40, -27, -13, 0, 9, 26, 32, 38, 46],
        labels = [
                    "Extreme Cold Stress",
                    "Very Strong Cold Stress",
                    "Strong Cold Stress",
                    "Moderate Cold Stress",
                    "Slight Cold Stress",
                    "No Thermal Stress",
                    "Moderate Heat Stress",
                    "Strong Heat Stress",
                    "Very Strong Heat Stress",
                    "Extreme Heat Stress"
                 ],
        colors = [
                    "#0D104B",
                    "#262972",
                    "#3452A4",
                    "#3C65AF",
                    "#37BCED",
                    "#2EB349",
                    "#F38322",
                    "#C31F25",
                    "#7F1416",
                    "#580002"
                 ],
    )

    UTCI_Warm = UTCIColorScheme(
        name = "Warm",
        levels = [-40, -27, -13, 0, 9, 26, 28, 32, 38, 46],
        labels = [
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
                    "Extreme Heat Stress"
                 ],
        colors = [
                    "#0D104B",
                    "#262972",
                    "#3452A4",
                    "#3C65AF",
                    "#37BCED",
                    "#2EB349",
                    "#D3D317",
                    "#F38322",
                    "#C31F25",
                    "#7F1416",
                    "#580002"
                 ],
    )
