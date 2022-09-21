import numpy as np


from ladybugtools_toolkit import analytics


@analytics
def proximity_decay(
    value: float,
    distance_to_value: float,
    max_distance: float,
    decay_method: str = "linear",
) -> float:
    """Calculate the "decayed" value based on proximity (up to a maximum distance).

    Args:
        value (float):
            The value to be distributed.
        distance_to_value (float):
            A distance at which to return the magnitude.
        max_distance (float):
            The maximum distance to which magnitude is to be distributed. Beyond this, the input
            value is 0.
        decay_method (str, optional):
            A type of distribution (the shape of the distribution profile). Defaults to "linear".

    Returns:
        float:
            The value at the given distance.
    """

    distance_to_value = np.interp(distance_to_value, [0, max_distance], [0, 1])

    if decay_method == "linear":
        return (1 - distance_to_value) * value
    elif decay_method == "parabolic":
        return (-(distance_to_value**2) + 1) * value
    elif decay_method == "sigmoid":
        return (1 - (0.5 * (np.sin(distance_to_value * np.pi - np.pi / 2) + 1))) * value
    else:
        raise ValueError(f"Unknown curve type: {decay_method}")
