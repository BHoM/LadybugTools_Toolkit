import numpy as np


def direction_bin_centers(
    directions: int = 36,
) -> list[float]:
    """Calculate the bin centers for a given number of directions.
    This returns a list the length of the number of directions, with each
    bin center representing the centroid of a directional bin. The first
    value is always 0 (or north), and centers move clockwise from there.

    Args:
        directions (int):
            The number of directions to calculate bin centers for.

    Returns:
        list[float]:
            A list of bin centers.

    """
    return np.linspace(0, 360, directions + 1)[:-1].tolist()


def direction_bin_edges(
    directions: int = 36,
) -> list[float]:
    """Calculate the bin edges for a given number of directions.
    The returned list includes half bins for the ranges about 0/360, so
    that the first and last pair of values the list are "half-bins".

    Args:
        directions (int):
            The number of directions to calculate bin edges for.

    Returns:
        list[float]:
            A list of bin edges.

    """
    bin_width = 360 / directions
    if bin_width == 360:
        bin_edges = np.array([0, 360])
    else:
        bin_edges = np.array(direction_bin_centers(directions=directions)) - (
            bin_width / 2
        )
        bin_edges = np.where(bin_edges < 0, 360 + bin_edges, bin_edges)
        bin_edges = np.append(bin_edges, bin_edges[0])
        bin_edges[0] = 0
        bin_edges = np.append(bin_edges, 360)
    return bin_edges.tolist()
