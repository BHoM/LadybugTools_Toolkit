import pandas as pd

def solar_radiation_metadata(values, directions, tilts):
    df = pd.DataFrame(values)
    df.index = tilts
    df.columns = directions
    return {
        "max_value": df.max().max(),
        "max_direction": df.max().idxmax(),
        "max_tilt": df.idxmax()[df.max().idxmax()],
        "min_value": df.min().min(),
        "min_direction": df.min().idxmin(),
        "min_tilt": df.idxmin()[df.min().idxmin()]
        }