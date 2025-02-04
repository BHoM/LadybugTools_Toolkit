from pathlib import Path

import pandas as pd
from ladybugtools_toolkit.abc.abc_model import ABCModel, Clothing, Phase
from pytest import approx

from ladybugtools_toolkit.abc.util import \
    run_abc

from .. import ABC_MODEL_FILE


def test_clothing_from_clo():
    """_"""
    clothing = Clothing.from_clo(0.5)
    assert isinstance(clothing, Clothing)
    assert clothing.estimate_clo() == approx(0.5)

def test_clothing_predefined():
    predefined_clothing_ensembles = [
        "nude",
        "summer_light",
        "summer_casual",
        "summer_business_casual",
        "winter_casual",
        "winter_business_formal",
        "winter_outerwear",
    ]
    for ensemble in predefined_clothing_ensembles:
        assert isinstance(getattr(Clothing, ensemble)(), Clothing)


def test_phase_good():
    clothing = Clothing.from_clo(0.5)
    phase = Phase(
        ta=25.0,
        mrt=25.0,
        rh=0.5,
        solar=200.0,
        v=0.1,
        met=1.2,
        met_activity_name="Sitting",
        clothing=clothing,
        start_time=0,
        ramp=False,
        end_time=60,
        time_units="minutes"
    )

    # Assert that the Phase object is created successfully with the correct attributes
    assert phase.ta == 25.0
    assert phase.mrt == 25.0
    assert phase.rh == 0.5
    assert phase.solar == 200.0
    assert phase.v == 0.1
    assert phase.met == 1.2
    assert phase.met_activity_name == "Sitting"
    assert phase.clothing == clothing
    assert phase.start_time == 0
    assert phase.ramp is False
    assert phase.end_time == 60
    assert phase.time_units == "minutes"

def test_phase_from_met_clo():
    """_"""
    phase = Phase.from_met_clo(
        ta=25.0,
        mrt=25.0,
        rh=0.5,
        solar=200.0,
        v=0.1,
        met=1.2,
        clo=0.5,
        start_time=0,
        ramp=False,
        end_time=60,
        time_units="minutes"
    )
    assert isinstance(phase, Phase)
    assert phase.clothing.estimate_clo() == approx(0.5)

def test_abc_model_from_csv():
    """_"""
    abc_model = ABCModel.from_csv(csv_file=ABC_MODEL_FILE, name="ABCModel_Test", description="A test ABC model")
    assert isinstance(abc_model, ABCModel)

def test_run_abc_model(tmpdir):
    """_"""
    abc_model = ABCModel.from_csv(csv_file=ABC_MODEL_FILE, name="ABCModel_Test", description="A test ABC model")
    
    # create file paths
    _dir = tmpdir.mkdir("abc_model")
    input_json_file = Path(_dir.join("input_json_file.json"))
    output_json_file = Path(_dir.join("output_json_file.json"))
    output_csv_file = Path(_dir.join("output_csv_file.csv"))

    # Run the ABC model
    abc_model.to_file(input_json_file)

    run_abc(
        input_json_file=input_json_file,
        output_json_file=output_json_file,
        output_csv_file=output_csv_file,
    )

    # read results into a dataframe
    results_df = pd.read_csv(output_csv_file)
    assert isinstance(results_df, pd.DataFrame)
    for col in [
        "elapsed_time",
        "Ta",
        "MRT",
        "RH",
        "Velocity",
        "Solar",
        "Clo",
        "Met",
        "Overall_Comfort",
        "Overall_Comfort_weighted",
        "Overall_Sensation",
        "Overall_Sensation_Linear",
        "Overall_Sensation_Weighted",
        "MeanSkinTemp",
        "Tblood",
        "Tneutral",
        "PMV",
        "PPD",
        "EHT",
        "Qmet",
        "Qconv",
        "Qrad",
        "Qsolar",
        "Qresp",
        "Qsweat",
        "Tskin-Head",
        "Tcore-Head",
        "Sens-Head",
        "Sens_weighted-Head",
        "Comfort-Head",
        "Comfort_weighted-Head",
        "EHT-Head",
        "Tskin_set-Head",
        "Tskin_set_reg-Head",
        "Tskin-Chest",
        "Tcore-Chest",
        "Sens-Chest",
        "Sens_weighted-Chest",
        "Comfort-Chest",
        "Comfort_weighted-Chest",
        "EHT-Chest",
        "Tskin_set-Chest",
        "Tskin_set_reg-Chest",
        "Tskin-Back",
        "Tcore-Back",
        "Sens-Back",
        "Sens_weighted-Back",
        "Comfort-Back",
        "Comfort_weighted-Back",
        "EHT-Back",
        "Tskin_set-Back",
        "Tskin_set_reg-Back",
        "Tskin-Pelvis",
        "Tcore-Pelvis",
        "Sens-Pelvis",
        "Sens_weighted-Pelvis",
        "Comfort-Pelvis",
        "Comfort_weighted-Pelvis",
        "EHT-Pelvis",
        "Tskin_set-Pelvis",
        "Tskin_set_reg-Pelvis",
        "Tskin-Left Upper Arm",
        "Tcore-Left Upper Arm",
        "Sens-Left Upper Arm",
        "Sens_weighted-Left Upper Arm",
        "Comfort-Left Upper Arm",
        "Comfort_weighted-Left Upper Arm",
        "EHT-Left Upper Arm",
        "Tskin_set-Left Upper Arm",
        "Tskin_set_reg-Left Upper Arm",
        "Tskin-Right Upper Arm",
        "Tcore-Right Upper Arm",
        "Sens-Right Upper Arm",
        "Sens_weighted-Right Upper Arm",
        "Comfort-Right Upper Arm",
        "Comfort_weighted-Right Upper Arm",
        "EHT-Right Upper Arm",
        "Tskin_set-Right Upper Arm",
        "Tskin_set_reg-Right Upper Arm",
        "Tskin-Left Lower Arm",
        "Tcore-Left Lower Arm",
        "Sens-Left Lower Arm",
        "Sens_weighted-Left Lower Arm",
        "Comfort-Left Lower Arm",
        "Comfort_weighted-Left Lower Arm",
        "EHT-Left Lower Arm",
        "Tskin_set-Left Lower Arm",
        "Tskin_set_reg-Left Lower Arm",
        "Tskin-Right Lower Arm",
        "Tcore-Right Lower Arm",
        "Sens-Right Lower Arm",
        "Sens_weighted-Right Lower Arm",
        "Comfort-Right Lower Arm",
        "Comfort_weighted-Right Lower Arm",
        "EHT-Right Lower Arm",
        "Tskin_set-Right Lower Arm",
        "Tskin_set_reg-Right Lower Arm",
        "Tskin-Left Hand",
        "Tcore-Left Hand",
        "Sens-Left Hand",
        "Sens_weighted-Left Hand",
        "Comfort-Left Hand",
        "Comfort_weighted-Left Hand",
        "EHT-Left Hand",
        "Tskin_set-Left Hand",
        "Tskin_set_reg-Left Hand",
        "Tskin-Right Hand",
        "Tcore-Right Hand",
        "Sens-Right Hand",
        "Sens_weighted-Right Hand",
        "Comfort-Right Hand",
        "Comfort_weighted-Right Hand",
        "EHT-Right Hand",
        "Tskin_set-Right Hand",
        "Tskin_set_reg-Right Hand",
        "Tskin-Left Thigh",
        "Tcore-Left Thigh",
        "Sens-Left Thigh",
        "Sens_weighted-Left Thigh",
        "Comfort-Left Thigh",
        "Comfort_weighted-Left Thigh",
        "EHT-Left Thigh",
        "Tskin_set-Left Thigh",
        "Tskin_set_reg-Left Thigh",
        "Tskin-Right Thigh",
        "Tcore-Right Thigh",
        "Sens-Right Thigh",
        "Sens_weighted-Right Thigh",
        "Comfort-Right Thigh",
        "Comfort_weighted-Right Thigh",
        "EHT-Right Thigh",
        "Tskin_set-Right Thigh",
        "Tskin_set_reg-Right Thigh",
        "Tskin-Left Lower Leg",
        "Tcore-Left Lower Leg",
        "Sens-Left Lower Leg",
        "Sens_weighted-Left Lower Leg",
        "Comfort-Left Lower Leg",
        "Comfort_weighted-Left Lower Leg",
        "EHT-Left Lower Leg",
        "Tskin_set-Left Lower Leg",
        "Tskin_set_reg-Left Lower Leg",
        "Tskin-Right Lower Leg",
        "Tcore-Right Lower Leg",
        "Sens-Right Lower Leg",
        "Sens_weighted-Right Lower Leg",
        "Comfort-Right Lower Leg",
        "Comfort_weighted-Right Lower Leg",
        "EHT-Right Lower Leg",
        "Tskin_set-Right Lower Leg",
        "Tskin_set_reg-Right Lower Leg",
        "Tskin-Left Foot",
        "Tcore-Left Foot",
        "Sens-Left Foot",
        "Sens_weighted-Left Foot",
        "Comfort-Left Foot",
        "Comfort_weighted-Left Foot",
        "EHT-Left Foot",
        "Tskin_set-Left Foot",
        "Tskin_set_reg-Left Foot",
        "Tskin-Right Foot",
        "Tcore-Right Foot",
        "Sens-Right Foot",
        "Sens_weighted-Right Foot",
        "Comfort-Right Foot",
        "Comfort_weighted-Right Foot",
        "EHT-Right Foot",
        "Tskin_set-Right Foot",
        "Tskin_set_reg-Right Foot",
    ]:
        assert col in results_df.columns
    assert len(results_df) == 80
    assert results_df.sum().sum() == approx(456946.8107957152, rel=0.001)
