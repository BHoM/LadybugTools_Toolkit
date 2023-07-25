import os
from dataclasses import dataclass, field

from honeybee_energy.run import run_idf, run_osw, to_openstudio_osw
from honeybee_energy.simulation.parameter import SimulationParameter
from ladybug.epw import EPW


@dataclass
class EnergySimulation:
    model: Union[Model, BoxModel]
    epw: Union[EPW, Path, str]
    folder: str = field(init=True, default= None)
    simulation_parameters: SimulationParameter = field(init=True, default= EnergySettings().simulation_parameters())

    def __post_init__(self):
        ### Set up
        # Converts str or Path to EPW obj
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)
                # Set unique identifier (shared with Box Model if that's given)
        
        if isinstance(self.model, BoxModel):
            self.identifier = self.model.identifier
            self.model = self.model.model # TODO this code looks ropey 
        else:
            self.identifier = str(uuid4())

        # duplicate input to avoid making changes to original model
        self.model = self.model.duplicate()
        self.simulation_parameters = self.simulation_parameters.duplicate() # ensure input is not edited

        # Set simulation folder
        if self.folder is None:
            self.folder = hb_config.folders.default_simulation_folder # TODO add the id to this? Needs to be checked

    def simulate(self):
        # TODO review this, copied some old code
        # get epw file path
        epw_file = self.epw.file_path
        hb_model = self.model
        clean_name = clean_name = re.sub(r'[^.A-Za-z0-9_-]', '_', hb_model.display_name)
        directory = os.path.join(self.folder, clean_name, 'openstudio')
        epw_obj = EPW(epw_file)
        silent = False

        # Prepare folders
        nukedir(directory, True)
        preparedir(directory)
        sch_directory = os.path.join(directory, 'schedules')
        preparedir(sch_directory)

        # Produce energy settings JSON
        # get energy settings and suplicate so original is unchanged
        energy_parameters = EnergySettings().simulation_parameters()
        energy_parameters = energy_parameters.duplicate()

        # add autocalculated design days
        des_days = [epw_obj.approximate_design_day('WinterDesignDay'),
                    epw_obj.approximate_design_day('SummerDesignDay')]
        energy_parameters.sizing_parameter.design_days = des_days

        # create json of energy parameters for iput to simulation
        energy_param_dict = energy_parameters.to_dict()
        energy_param_json = os.path.join(directory, 'simulation_parameter.json')
        with open(energy_param_json, 'w') as fp:
            json.dump(energy_param_dict, fp)

        # Produce model parameters JSON
        model_dict = hb_model.to_dict(triangulate_sub_faces=True)
        hb_model.properties.energy.add_autocal_properties_to_dict(model_dict)
        model_json = os.path.join(directory, '{}.hbjson'.format(clean_name))
        with open(model_json, 'w') as fp:
            json.dump(model_dict, fp)

        # Run simulation
        osw_path = to_openstudio_osw(model_path = model_json, osw_directory = directory, sim_par_json_path = energy_param_json)
        osm, idf = run_osw(osw_json= osw_path, silent=silent)
        sql, zsz, rdd, html, err = run_idf(idf_file_path = idf, epw_file_path = epw_file, silent=silent)

        return sql