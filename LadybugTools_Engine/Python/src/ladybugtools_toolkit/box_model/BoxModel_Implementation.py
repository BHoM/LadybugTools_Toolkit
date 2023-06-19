from ladybugtools_toolkit.prototypes.BoxModel.box_model import BoxModel

box_model = BoxModel()._create_hb_model()

submit -> visualise the model - visual check of set up 
simulate -> simulates both energy and daylight and returns all results

simulate:
model_setup
simulate_box_model
results processer 
returns: dict of results, filepath to images, filepath to results vtk (including mesh results)

def model_setup(BoxModel, HVAC_obj, Construction_obj) -> HB_Model:
    box_model.properties.energy.set_HVAC(HVAC_obj)
    box_model.properties.xxxx
    return box_model

def simulate_box_model(energy = True, daylight = True) -> Results_Obj:
    box_model.simulate_energy()
    box_model.simulate_daylight()
    return Results_Obj

@dataclass
class Results_Obj:
    energy_summary_metrics: dict:
        {solar gain: XX
         heating : xx
         }

def plot_energy_reslts(Result_Obj):
    returns plots


# Results obj to database 

def 

IdealAir_dict = {
    "type": "IdealAirSystem",
    "identifier": "Classroom1 Ideal Loads Air System",  # identifier for the HVAC
    "display_name": "Standard IdealAir",  # name for the HVAC
    "economizer_type": 'DifferentialDryBulb',  # Economizer type
    "demand_controlled_ventilation": False,  # Demand controlled ventilation
    "sensible_heat_recovery": 0,  # Sensible heat recovery effectiveness
    "latent_heat_recovery": 0,  # Latent heat recovery effectiveness
    "heating_air_temperature": 50,  # Heating supply air temperature
    "cooling_air_temperature": 13,  # Cooling supply air temperature
    "heating_limit": {'type': 'NoLimit'},  # Max size of the heating system
    "cooling_limit": {'type': 'NoLimit'},  # Max size of the cooling system
    "heating_availability": {},  # Schedule for availability of heat or None
    "cooling_availability": {}  # Schedule for availability of cooling or None
    }
