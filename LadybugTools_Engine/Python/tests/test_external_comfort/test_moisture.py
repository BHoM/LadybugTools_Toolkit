from pathlib import Path
from tempfile import gettempdir
import pytest

import numpy as np

from ladybugtools_toolkit.external_comfort.spatial.moisture import (
    MoistureSource,
    DecayMethod,
    Point2D,
    spatial_evaporative_cooling_effect,
)
from .. import EPW_OBJ

EVAP_CLG_EFFECT_0 = np.where(np.arange(8760) % 2 == 0, 0.1, 0.9)
EVAP_CLG_EFFECT_1 = np.where(np.arange(8760) % 5 == 0, 0.9, 0.0)

n = 101
x = np.linspace(-50, 50, 11)
y = np.linspace(-50, 50, 11)
xx, yy = np.meshgrid(x, y)
POINTS = np.concatenate(np.stack([xx, yy], axis=2))
SOURCE = POINTS[60]


def test_round_trip():
    """_"""
    test_moisture_source = MoistureSource(
        identifier="example",
        evaporative_cooling_effect=EVAP_CLG_EFFECT_0,
        decay_method=DecayMethod.LINEAR,
        max_decay_distance=25,
        plume_angle=35,
    )

    tempfile = Path(gettempdir()) / "pytest_moisture_source.json"
    MoistureSource.from_dict(test_moisture_source.to_dict())
    MoistureSource.from_json(test_moisture_source.to_json())
    MoistureSource.from_file(test_moisture_source.to_file(tempfile))

    tempfile.unlink()


def test_moisture_source_bad():
    """_"""

    # non 8760 length evap clg effect
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[1],
            decay_method=DecayMethod.LINEAR,
            max_decay_distance=25,
            plume_angle=35,
        )

    # non numeric evap clg effect
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=["hi"] * 8760,
            decay_method=DecayMethod.LINEAR,
            max_decay_distance=25,
            plume_angle=35,
        )

    # negative evap clg effect
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[-1] * 8760,
            decay_method=DecayMethod.LINEAR,
            max_decay_distance=25,
            plume_angle=35,
        )

    # >1 evap clg effect
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[1.1] * 8760,
            decay_method=DecayMethod.LINEAR,
            max_decay_distance=25,
            plume_angle=35,
        )

    # bad decay method
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[1.1] * 8760,
            decay_method="not a decay method",
            max_decay_distance=25,
            plume_angle=35,
        )

    # bad max_decay_distance
    with pytest.raises(ValueError):
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[1.1] * 8760,
            decay_method=DecayMethod.LINEAR,
            max_decay_distance=0,
            plume_angle=35,
        )
        MoistureSource(
            identifier="example",
            evaporative_cooling_effect=[1.1] * 8760,
            decay_method=DecayMethod.LINEAR,
            max_decay_distance="yo",
            plume_angle=35,
        )


def test_spatial_evaporative_cooling_effect():
    """_"""
    test_moisture_source = MoistureSource(
        identifier="example",
        evaporative_cooling_effect=EVAP_CLG_EFFECT_0,
        decay_method=DecayMethod.LINEAR,
        max_decay_distance=25,
        plume_angle=35,
    )
    df = test_moisture_source.spatial_evaporative_cooling_effect(
        epw=EPW_OBJ, source=SOURCE, points=POINTS
    )
    assert df.values.shape == (8760, 121)
    assert df.values.sum() == pytest.approx(4090.620731753589, rel=0.001)
    assert df.values.mean() == pytest.approx(0.003859221793042746, rel=0.0001)


def test_composite_spatial_evaporative_cooling_effect():
    """_"""
    test_moisture_source_0 = MoistureSource(
        identifier="example_0",
        evaporative_cooling_effect=EVAP_CLG_EFFECT_0,
        decay_method=DecayMethod.LINEAR,
        max_decay_distance=25,
        plume_angle=35,
    )
    test_moisture_source_1 = MoistureSource(
        identifier="example_1",
        evaporative_cooling_effect=EVAP_CLG_EFFECT_1,
        decay_method=DecayMethod.PARABOLIC,
        max_decay_distance=35,
        plume_angle=25,
    )

    df = spatial_evaporative_cooling_effect(
        moisture_sources=[test_moisture_source_0, test_moisture_source_1],
        epw=EPW_OBJ,
        points=POINTS,
        sources=[Point2D(15, 15), Point2D(0, 0)],
    )
    assert df.values.shape == (8760, 121)
    assert df.values.sum() == pytest.approx(1610.3179, rel=0.001)
    assert df.values.mean() == pytest.approx(0.0015192252, rel=0.0001)
