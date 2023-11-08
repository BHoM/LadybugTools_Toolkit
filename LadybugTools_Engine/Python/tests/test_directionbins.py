from pathlib import Path
from tempfile import gettempdir
import numpy as np
import pytest
from ladybugtools_toolkit.wind import DirectionBins

TEST_DIRECTION_BINS = DirectionBins()


def test_direction_bins():
    """."""
    db = DirectionBins()
    assert db.bin_width == 45
    assert db.directions == 8
    assert db.bins.sum() == 3240
    assert db.cardinal_directions == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    with pytest.raises(ValueError):
        db.bin_data(
            direction_data=np.linspace(0, 360, 100), other_data=np.linspace(20, 30, 50)
        )
    assert (
        sum(db.prevailing(direction_data=np.linspace(0, 360, 100), n=2, as_angle=True))
        == 135
    )
    assert db.prevailing(
        direction_data=np.linspace(0, 360, 100), n=2, as_angle=False
    ) == ["N", "SE"]

    db = DirectionBins(directions=16)
    assert db.bin_width == 22.5
    assert db.directions == 16
    assert db.bins.sum() == 6120
    assert db.cardinal_directions == [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    db = DirectionBins(directions=7, centered=False)
    assert db.bin_width == pytest.approx(51.42857142857143, rel=0.01)
    assert db.directions == 7
    assert db.bins.sum() == 2520.0
    assert db.cardinal_directions == ["NNE", "EbN", "SEbE", "S", "SWbW", "WbN", "NNW"]

    with pytest.raises(ValueError):
        DirectionBins.direction_bin_edges(directions=1, centered=False)


def test_round_trip():
    """_"""
    tempfile = Path(gettempdir()) / "pytest_directionbins.json"
    DirectionBins.from_dict(TEST_DIRECTION_BINS.to_dict())
    DirectionBins.from_json(TEST_DIRECTION_BINS.to_json())
    DirectionBins.from_file(TEST_DIRECTION_BINS.to_file(tempfile))

    with pytest.raises(ValueError):
        TEST_DIRECTION_BINS.to_file("./not_a_json_file.txt")

    tempfile.unlink()
