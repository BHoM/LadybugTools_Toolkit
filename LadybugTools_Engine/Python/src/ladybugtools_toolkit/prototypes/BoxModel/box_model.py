import json
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from honeybee.boundarycondition import boundary_conditions
from honeybee.config import folders as hb_folders
from honeybee.face import Face
from honeybee.facetype import Wall
from honeybee.model import Model, Room
from honeybee_energy.constructionset import ConstructionSet
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.programtype import ProgramType
from honeybee_radiance.modifierset import ModifierSet
from honeybee_radiance.sensorgrid import SensorGrid
from honeybee_vtk.model import Model as VTKModel
from ladybug_geometry.geometry3d.pointvector import Point3D

"""
Within class BoxModel:
- Create the model: Box_Model
Room: Box_Room

"""

@dataclass
class BoxModel:
    # geometry
    name: str = field(init=True, default=None)
    bay_width: float = field(init=True, default=3)
    count_bays: int = field(init = True, default=3)
    height: float = field(init=True, default=3)
    depth: float = field(init=True, default=10) 
    orientation_angle: float = field(init= True, default = 0)

    # geometry - glazing
    # TODO add glazing ratio per facade
    glazing_ratio: float = field(init= True, default = 0.4)
    # targets may not be achieved, LBT will overide if needed to meet glazing ratio - TODO raise warning if not met?  
    target_window_height: float = field(init=True, default=2)
    target_sill_height: float = field(init=True, default=0.8)

    #geometry - daylight
    wall_thickness: float = field(init=True, default = 0.5)
    sensor_grid_size: float = field(init=True, default = 0.2)
    sensor_grid_offset: float = field(init=True, default = 0.8)
    sensor_grid_bay_count: int = field(init=True, default = 1)

    # setup - energy
    # TODO test impact of NoLimit vs Autosize, assume IES equivalent is NoLimit. Below is using autosize
    hvac_config: IdealAirSystem = field(init=True, default=IdealAirSystem(identifier="generic_hvac",
                                                                          economizer_type="NoEconomizer",
                                                                          demand_controlled_ventilation= False))
    # TODO likely need to separate out a BoxModel_ConstructionSet that gives adjusting wall and window U-value / g-value 
    construction_config: ConstructionSet = field(init=True, default=ConstructionSet(identifier="generic_construction"))
    # TODO allow changing aspects of program type, occupancy, lighting, equipment
    program_config: ProgramType = field(init=True, default = ProgramType(identifier="generic_program"))

    # setup - radiance
    # TODO allow for custom reflactances and transmittance
    modifier_config: ModifierSet = field(init=True, default=ModifierSet(identifier="generic_modifier"))

    def __post_init__(self):
        # Calculated variables
        # total box model width
        self.width = self.bay_width*self.count_bays
        # origin at center
        self.origin = Point3D(x=-self.width/2, y=-self.depth/2, z=0)

        # Error checking
        # check that height is between 1 and 10
        if not 1 <= self.height <= 10:
            raise ValueError("Height must be between 1 and 10.")
        # check glazing ratio is between 0 and 0.95
        if not self.glazing_ratio <= 0.95:
            raise ValueError("Glazing ratio must be less than 0.95")
        if self.sensor_grid_bay_count > self.count_bays:
            raise ValueError("sensor_grid_bay_count must be less than or equal to count_bays")       

        # automated naming if none given
        if self.name is None:
            self.name = f"Generic_Box_Model_W{self.width:0.0f}"

    def _create_room(self) -> Room:
        room = Room.from_box(identifier = self.name, width = self.width, height = self.height, origin=self.origin)
        for face in room.faces:
            face.boundary_condition = boundary_conditions.adiabatic
        room.faces[1].boundary_condition = boundary_conditions.outdoors
        room.faces[1].apertures_by_ratio_rectangle(ratio =self.glazing_ratio, aperture_height =self.target_window_height,
                                                  sill_height =self.target_sill_height, horizontal_separation= self.bay_width)
        return room


    def _create_hb_model(self) -> Model:
        room = self._create_room()
        return Model(identifier = self.name, rooms = [room])
    
    @property
    def model(self) -> Model:
        return self._create_hb_model()
    
    def to_html(self):
        model = self._create_hb_model()
        return VTKModel(model).to_html(name = self.name, show = True)
    
    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        raise NotImplementedError("create BoxModel from dict")
    
    @classmethod
    def from_json(cls, json_string: str):
        """Create BoxModel from a JSON string.
        
        Args:
            json_string (str):
                This is a string containing stuff for making a HB Model.

        Returns:
            BoxModel:
                Returns a BoxModel
        """
        
        dictionary = json.loads(json_string)

        return cls.from_dict(dictionary)


###########################


@dataclass
class Results:
    """Base class for results from a simulation run.

    Args:
        results_file: Ladybug sim folder for object simuialted
    """

    results_file: Path


@dataclass
class EnergyResults(Results):
    heating_load: pd.Series = field(init=True, repr=False, compare=False)
    cooling_load: pd.Series = field(init=True, repr=False, compare=False)
    solar_gain: pd.Series = field(init=True, repr=False, compare=False)

    def __post_init__(self):
        if not self.results_file.exists():
            raise ValueError("Results directory does not exist!")
        # e+ results exist
        if not (self.results_file / "eplusout.sql").exists():
            raise ValueError("Results directory does not contain an eplusout.sql file!")

        # load in the results - EXAMPLE!
        raise NotImplementedError("Load Energy results from an SQL file.")

    @property
    def peak_heating_load(self):
        return self.heating_load.max()

    @property
    def peak_cooling_load(self):
        return self.cooling_load.max()

    @property
    def total_load(self):
        return self.heating_load + self.cooling_load

    @property
    def peak_total_load(self):
        return self.total_load.max()

    @property
    def peak_solar_gain(self):
        return self.solar_gain.max()

    def plot_load_balance(self) -> plt.Figure:
        raise NotImplementedError()
        fig, ax = plt.subplots()
        ax.plot(self.heating_load, label="Heating")
        ax.plot(self.cooling_load, label="Cooling")
        ax.plot(self.total_load, label="Total")
        ax.legend()
        return fig


@dataclass
class DaylightResults(Results):
    illuminance: pd.DataFrame = field(init=True, repr=False, compare=False)

    def __post_init__(self):
        if not self.results_file.exists():
            raise ValueError("Results directory does not exist!")
        # annual illuminace results exist
        if not (self.results_file / "annual_illuminance.csv").exists():
            raise ValueError(
                "Results directory does not contain an annual_illuminance.csv file!"
            )

        # load in the results - EXAMPLE!
        raise NotImplementedError("Load illumiancne results from an ILL file.")

    def plot_daylight_autonomy(self) -> plt.Figure:
        raise NotImplementedError()

    def get_illuminance(self) -> pd.DataFrame:
        """Returns a time-indexed DF with point-columns ILL values."""
        raise NotImplementedError()


###########################
# SIMULATION #


class SimulationType(Enum):
    ENERGY = auto()
    DAYLIGHT = auto()
    THERMAL_COMFORT = auto()


@dataclass
class BoxModelSimulation:
    box_model: Model = field(init=True, repr=True, compare=True)
    epw_file: Path = field(init=True, repr=True, compare=True)

    simulation_options: List[SimulationType] = field(
        init=True,
        repr=True,
        compare=True,
        default_factory=lambda: [SimulationType.ENERGY, SimulationType.DAYLIGHT],
    )

    _eplus_results_file: Path = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not Path(self.epw_file).exists():
            raise ValueError("EPW file does not exist!")
        if not isinstance(self.box_model, Model):
            raise ValueError("Box model must be a honeybee model!")

        _eplus_results_file = (
            Path(hb_folders.default_simulation_folder)
            / self.box_model.identifier
            / "run"
            / "eplusout.sql"
        )

    def _already_run(self, sim_type: SimulationType) -> bool:
        # check if the simulation has already been run
        # return True or False
        raise NotImplementedError()

    def run_energy(self) -> EnergyResults:
        # Run an energy simulation using Honeybee energy

        # check if it's alresady run, and retrun existign results if True
        if self._already_run(SimulationType.ENERGY):
            return EnergyResults(results_file=self._eplus_results_file)

        # get the sql file output
        # load the sql file
        # return the results
        # return EnergyResults(results_file=sql_file)
        raise NotImplementedError()

    def run_daylight(self) -> EnergyResults:
        # Run a daylight simulation using Honeybee radiance

        raise NotImplementedError()

    def run_thermal_comfort(self) -> EnergyResults:
        # Run a daylight simulation using Honeybee radiance
        raise NotImplementedError()

    def run(self) -> Results:
        energy_results = self.run_energy()
        daylight_results = self.run_daylight()


def main():
    print(BoxModel(name = "Test", target_sill_height= 0, target_window_height= 3, glazing_ratio= 0.8).to_html())


if __name__ == "__main__":
    main()
