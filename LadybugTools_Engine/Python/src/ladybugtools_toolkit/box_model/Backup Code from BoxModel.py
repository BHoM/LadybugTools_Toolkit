# @dataclass
# class FabricAttributes(ConstructionSet):
#     # input same propertues as you would in construction set
#     # inide the init functions modifty these based on what you want
#     def __post_init__(self):
#         constr: List[WindowConstruction] = self.aperture_set.constructions[0]

@dataclass
class FabricAttribute:
    epw: Union[EPW, Path, str]
    base_vintage: str = field(init=True, default = '2019')
    base_construction_type: str = field(init=True, default = 'SteelFramed')

    def __post_init__(self):
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)
        
        base_constr_set = constr_set_from_base_climate_vintage(new_set_identifier= self.name+ "_"+ self.identifier,
                                                epw = self.epw, vintage= self.base_vintage,
                                                construction_type= self.construction_type)
        
        new_constr_set = base_constr_set.duplicate()
        new_constr_set.unlock()

        self.constr_set: ConstructionSet = new_constr_set
    
    @staticmethod
    def adjust_u_factor(construction: OpaqueConstruction, target_u_factor: float):
        r1 = construction.r_factor
        r2 = construction.materials[2].r_value
        r_remainder = r1 - r2
        new_r = 1/target_u_factor - r_remainder
        construction.materials[2].r_value = new_r
        return construction

@dataclass
class ExternalWallAttribute(OpaqueFabricAttribute):
    u_value: float
    wall_int_vis_reflectance: float
    wall_int_sol_absorptance: float
    wall_ext_sol_absorptance: float

    def __post_init__(self):
        
    
    @property
    def external_wall_construction(self) -> OpaqueConstruction:
        self.constr_set.external


@dataclass
class GlazedFabricAttribute(FabricAttribute):
    u_value: float  =field()
    g_value: float  =field()
    light_transmittance: float = field()


    def default_construction()
        #get fdefault of type and unlock. 

    @staticmethod
    def adjust_int_reflectance():
        #do the thing

@dataclass
class WindowFabcirAttributes:
    epw: Union[EPW, Path, str]
    base_vintage: str = field(init=True, default = '2019')
    base_construction_type: str = field(init=True, default = 'SteelFramed')

    u_value: float  =field()
    g_value: float  =field()
    ltv: float = field()

    @property
    def hb_construction():
        # method to 
        return WindowConstruction()

@dataclass
class EnvelopeAttr:
    epw: EPW
    wall_u_value: float
    wall_vis_refl: float
    window_g_value: float

    def __post_init__(self):
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)

        if self.wall_vis_refl > 1 or self.wall_vis_refl < 0:
            raise ValueError()
        self.wall_constr = WallFabcirAttributes(self.epw, self.wall_u_value, self.wall_vis_refl)

    @property
    def construction_set(self):
        # method here to buikld constr set from inputs
        return ConstructionSet("fasdfopisjadf", wall_set=[self.wall_constr.hb_construction, self.wall_constr.hb_construction, self.wall_constr.hb_construction])
    

@dataclass
class FabricAttributesXXX:
    epw: Union[EPW, Path, str]
    base_vintage: str = field(init=True, default = '2019')
    construction_type: str = field(init=True, default = 'SteelFramed')
    name: str = field(init=True, default = 'BM_Constr_Set')

    wall_u_value: float = field(init=True, default = 0.25)
    window_u_value: float = field(init=True, default = 1.4)
    window_g_value: float = field(init=True, default = 0.4)
    window_light_transmittance: float = field(init=True, default = 0.65)
    wall_visible_reflectance: float = field(init=True, default = 0.7)
    wall_solar_absorptance: float = field(init=True, default = 0.4)
    floor_visible_reflectance: float = field(init=True, default = 0.7)
    floor_solar_absorptance: float = field(init=True, default = 0.4)
    ceiling_visible_reflectance: float = field(init=True, default = 0.8)
    ceiling_solar_absorptance: float = field(init=True, default = 0.3)

    def __post_init__(self):


        # convert the input epw thingy into an epw object
        if isinstance(self.epw, (str, Path)):
            self.epw = EPW(self.epw)
        
        # Manually creaet wall set, aperture set, ground set, ....
        

        # Set up constructions
        self.constr_set: ConstructionSet = constr_set_from_base_climate_vintage(new_set_identifier= self.name+ "_"+ self.identifier,
                                                          epw = self.epw, vintage= self.base_vintage,
                                                          construction_type= self.construction_type)

        # External wall property adjustment
        bm_ext_wall = self.constr_set.wall_set.exterior_construction
        bm_ext_wall.unlock()

        self.adjust_u_factor(bm_ext_wall, target_u_factor=self.wall_u_value)
        adjust_inside_vis_reflectance(bm_ext_wall, target_reflectance= self.wall_visible_reflectance)
        adjust_inside_sol_absorptance(bm_ext_wall, target_absorptance= self.wall_solar_absorptance)

        self.constr_set.wall_set.exterior_construction = bm_ext_wall

        # Internal walls property adjustment
        bm_int_wall = self.constr_set.wall_set.interior_construction
        bm_int_wall.unlock()

        adjust_both_sides_vis_reflectance(bm_int_wall, target_reflectance= self.wall_visible_reflectance)
        adjust_both_sides_sol_absorptance(bm_int_wall, target_absorptance= self.wall_solar_absorptance)

        self.constr_set.wall_set.interior_construction = bm_int_wall

        # Floor property 
        bm_floor = self.constr_set.floor_set.interior_construction
        bm_floor.unlock()

        adjust_inside_vis_reflectance(bm_floor, target_reflectance= self.floor_visible_reflectance)
        adjust_inside_sol_absorptance(bm_floor, target_absorptance= self.floor_solar_absorptance)

        self.constr_set.floor_set.interior_construction = bm_floor

        # Ceiling property
        bm_ceiling = self.constr_set.roof_ceiling_set.interior_construction
        bm_ceiling.unlock()

        adjust_inside_vis_reflectance(bm_ceiling, target_reflectance= self.ceiling_visible_reflectance)
        adjust_inside_sol_absorptance(bm_ceiling, target_absorptance= self.ceiling_solar_absorptance)

        self.constr_set.roof_ceiling_set.interior_construction = bm_floor

        # External window property adjustment
        bm_window_material = EnergyWindowMaterialSimpleGlazSys(identifier='BM_window_material'+self.identifier,
                                                               u_factor=self.window_u_value,
                                                               shgc=self.window_g_value,
                                                               vt=self.window_light_transmittance)
        self.constr_set.aperture_set.window_construction = WindowConstruction(identifier='BM_window_construction'+self.identifier,
                                                                         materials=[bm_window_material],
                                                                         frame = None)

        modifier_set = self.constr_set.to_radiance_visible_interior()

        return self.constr_set, modifier_set

    # def _modify_wall_properties(self) -> None:
    #     const_set = self.constr_set

    #     const_set.exterior_construction
    #     bm_ext_wall.unlock()

    #     adjust_u_factor(bm_ext_wall, target_u_factor=self.wall_u_value)
    #     adjust_inside_vis_reflectance(bm_ext_wall, target_reflectance= self.wall_visible_reflectance)
    #     adjust_inside_sol_absorptance(bm_ext_wall, target_absorptance= self.wall_solar_absorptance)

    #     constr_set.wall_set.exterior_construction = bm_ext_wall

    @staticmethod
    def adjust_u_factor(construction: OpaqueConstruction, target_u_factor: float):
        r1 = construction.r_factor
        r2 = construction.materials[2].r_value
        r_remainder = r1 - r2
        new_r = 1/target_u_factor - r_remainder
        construction.materials[2].r_value = new_r
        return construction
    
# Not currently used
class GasType(str, Enum):
    Air = 'Air'
    Argon = 'Argon'
    Krypton = 'Krypton'
    Xenon = 'Xenon'

# Not currently used
@dataclass
class GlazingConstruction:
    number_of_panes: int = field(init=True, default = 2)
    gas_type: GasType = field(init=True, default = GasType.Argon)


@dataclass
class BoxModelSimulatable:
    box_model: BoxModel = field(init=True, default=BoxModel(epw= ))
    fabric_attributes: FabricAttributes = field(init=True, default=FabricAttributes(box_model.epw))
    # TODO add functionality to use different HVAC systems
    # TODO test impact of NoLimit vs Autosize, assume IES equivalent is NoLimit. Below is using autosize
    hvac_config: IdealAirSystem = field(init=True, default=IdealAirSystem(identifier="generic_hvac",
                                                                          economizer_type="NoEconomizer",
                                                                          demand_controlled_ventilation= False))
    # TODO allow changing aspects of program type, occupancy, lighting, equipment
    program_config: ProgramType = field(init=True, default = ProgramType(identifier="generic_program"))

    def __post_init__(self):
        model = BoxModel

def box_model_simulation_setup(box_model = BoxModel, program_type = ProgramType, hvac = None, modifier_set = ModifierSet) -> BoxModelSimulatable:
    pass


