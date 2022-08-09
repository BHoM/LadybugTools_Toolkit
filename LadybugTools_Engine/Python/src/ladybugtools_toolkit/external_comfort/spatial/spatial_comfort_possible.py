from pathlib import Path


def spatial_comfort_possible(simulation_directory: Path) -> bool:
    """Checks whether spatial_comfort processing is possible for a given simulation_directory.

    Args:
        simulation_directory (Path):
            A folder containing Honeybee-Radiance Sky-View and Annual Irradiance results.

    Returns:
        bool:
            True if possible. If impossible, then an error is raised instead.
    """

    simulation_directory = Path(simulation_directory)

    # Check for annual irradiance data
    annual_irradiance_directory = simulation_directory / "annual_irradiance"
    if (
        not annual_irradiance_directory.exists()
        or len(list((annual_irradiance_directory / "results").glob("**/*.ill"))) == 0
    ):
        raise FileNotFoundError(
            f"Annual-irradiance data is not available in {annual_irradiance_directory}."
        )

    # Check for sky-view data
    sky_view_directory = simulation_directory / "sky_view"
    if (
        not (sky_view_directory).exists()
        or len(list((sky_view_directory / "results").glob("**/*.res"))) == 0
    ):
        raise FileNotFoundError(
            f"Sky-view data is not available in {sky_view_directory}."
        )

    res_files = list((sky_view_directory / "results").glob("*.res"))
    if len(res_files) != 1:
        raise ValueError(
            "This process is currently only possible for a single Analysis Grid - multiple files found."
        )

    return True
