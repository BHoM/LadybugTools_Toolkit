from pathlib import Path


def cfd_directory(simulation_directory: Path) -> Path:
    """Get the CFD directory for a spatial simulation.

    Args:
        simulation_directory (Path):
            The associated simulation directory

    Returns:
        Path:
            The path to the moisture directory
    """

    if not (simulation_directory / "cfd").exists():
        raise FileNotFoundError(
            f'No "cfd" directory exists in {simulation_directory}. For this method to work, '
            + "you need a moisture directory containing a set of csv files extracted from CFD "
            + "simulations of at least 8 wind directions. Values in these files should correspond "
            + "with the wind velocities at the points from teh SpatialComfort case being assessed."
            + "\nFor example, the folder should contain 8 CSV files:"
            + '\n    ["./V315.csv", "./V000.csv", "./V045.csv", "./V090.csv", "./V135.csv", '
            + '"./V180.csv", "./V225.csv", "./V270.csv"]'
            + "\n... each containing point-velocities for the points in the SpatialComfort simulation"
            + "\nAdditionally, a JSON config should also be included, which stores the velocity "
            + "applied across the simulations for scaling in the thermal comfort assessment. An "
            + "example config can be found in this modules __init__.py"
        )

    return simulation_directory / "cfd"
