import numpy as np
from .ladybug_extension.epw import wind_speed_at_height
from ladybug.psychrometrics import saturated_vapor_pressure, wet_bulb_from_db_rh

def equibtemp(
    t_sky: float, # sky temperature (C)
    depth: float,  # depth (m)
    u: float,  # wind speed (m s-1)
    t_air: float,  # air temperature (C)
    t_wb: float, #wet bulb temperature (C)
    rh: float,  # relative humidity (%)
    q_solar: float,  # incoming solar energy (W)
    t_prev: float,  # previous water temperature (C)
    wind_height: float=10,  # wind height above water (m)
    albedo: float=0.08,  # albedo
    tstep: int=1
) -> float:
    """
    Subroutine to calculate the evaporation, using data from a
    water body using the equilibrium temperature model of de Bruin,
    H.A.R., 1982, j.hydrol, 59, 261-274

    Args:
        t_sky (float): Sky temperature (deg.C)
        depth (float): Depth of the water body (m)
        u (float): Wind speed (m s-1)
        t_air (float): Air temperature (deg.C)
        t_wb (float): Wet bulb temperature (deg.C)
        rh (float): Relative humidity (%)
        q_solar (float): Incoming solar radiation (W m-2)
        t_prev (float): Water temperature on previous time step (deg.C)
        wind_height (float): Height of wind measurements above water (m)
        albedo (float): Albedo of the water body
        tstep (int): Length of the time step (seconds)

    Returns:
        evap (float): Evaporation amount calculated using penman-monteith (kg lost over the time step)
    """
    # TODO - longer variable names
    # TODO - be clever about albedo based on angle of incidence of sun on water surface. if sun is normal to surface, albedo is low, if sun is low, albedo is high-er

    # conversions to SI units (J, kg, K, m)
    sigma = 5.7e-8  # stefan boltzmann constant
    k = 0.41  # Von Karman constant
    z0 = 0.001  # roughness length assumed to be very low
    ut = max(u, 0.01)  # wind speed (m s-1)
    zr = wind_height  # wind height above water (m)
    e_air = 0.70  # air emmissivity (assumption to be refined)
    e_w = 0.95  # water emmissivity
    lhv = alambdat(t_air)  # latent heat of vaporisation of water (J kg-1)
    t_sky += 273.15 # sky temperature (K)
    t_air += 273.15  # air temperature (K)
    t_wb += 273.15  # wet bulb temperature (K)
    t0 = t_prev + 273.15  # previous water temperature (K)
    cw = 4200  # specific heat capacity of water (J kg-1 K-1)
    c_a = 1013  # specific heat capacity of air (J kg-1 K-1)
    gamma = psyconst(100.0, lhv/(1000*1000))  # psychrometric constant (kPa K-1)
    rho_w = 1000  # density of water (kg m-3)
    rho_a = 1  # density of air (kg m-3)

    # NOTE - The magic numbers are magic. Trust them and they shall do you well. Might need some investigation, but for now the Env Agency says it's good
    wf = 4.4 + 1.82 * ut # W m-2 kPa-1

    # heat echange with the atmosphere/sky
    # uses sky temperature - https://www.engineeringtoolbox.com/radiation-heat-transfer-d_431.html Eqn3

    lwav_out_wb = e_w * (sigma * ((t_sky**4) - (t_wb**4)))  # W

    lwav_out_t0 = e_w * (sigma * ((t_sky**4) - (t0**4)))  # W

    # Net radiation for wet bulb or t0(previous temperature)
    net_rad_wb = (q_solar * (1 - albedo) + (lwav_out_wb))  # W

    net_rad_t0 = (q_solar * (1 - albedo) + (lwav_out_t0))  # W

    # time constant ... does something ... probably clever
    tau = (rho_w * cw * depth) / (
        4 * sigma * (t_wb**3) + wf * (delcalc(t_wb - 273.15) + gamma)
    )  # seconds

    # equilibrium temperature. Temperature at which no heat exchange occurs
    t_equib = t_wb + ((net_rad_wb) / (
        4 * sigma * (t_wb**3) + wf * (delcalc(t_wb - 273.15) + gamma)
    ))  # K

    # actual tempertaure of water after time step
    t_final = t_equib + (t0 - t_equib) * np.exp(tstep / tau)  # K

    # TODO - add additional heat gains in here, e.g. occupants, convection, conduction

    # net radiation exhange, minus evaporation energy loss (solar + longwave)
    N = -(rho_w * cw * depth * (t_final - t0))/tstep # W

    # aerodynamic resistance, another magic function
    ra = (1/15) * np.log(zr / z0) ** 2 / (k * k * ut) # m-1

    # latent heat flux (using penman-monteith)
    lambda_e = (
        delcalc(t_air - 273.15) * ((net_rad_t0) - (N))
        + rho_a * c_a * (vpdcalc(saturated_vapor_pressure(t_air), rh) / ra)
    ) / (
        delcalc(t_air - 273.15) + gamma
    )  # J
    
    # evaporation rate
    evap = (lambda_e) / lhv  # kg
    
    return evap

def delcalc(ta: float) -> float:
    """Function to calculate the slope of the vapour pressure curve

    Args:
        ta (float): air temperature (deg. C)

    Returns:
        float: slope of the vapour pressure curve (kPa deg. C-1)
    """

    ea = 0.611 * np.exp(17.27 * ta / (ta + 237.15))
    return 4099 * ea / (ta + 237.15) ** 2


def alambdat(t: float) -> float:
    """Function to correct the latent heat of vaporisation for temperature

    Args:
        t (float): temperature (deg. C)

    Returns:
        float: latent heat of vaporisation (J kg-1)
    """

    return (2.501 - t * 2.2361e-3) * 1000 * 1000


def psyconst(p: float, alambda: float) -> float:
    """Function to calculate the psychrometric constant from atmospheric pressure and latent heat of vaporisation

    Args:
        p (float): atmospheric pressure (kPa)
        alambda (float): latent heat of vaporisation (MJ kg-1)

    Returns:
        float: psychrometric constant (kPa deg. C-1)
    """

    cp = 1.013
    eta = 0.622
    return cp * p / (eta * alambda) * 1.0e-3


def vpdcalc(svp: float, rh: float) -> float:
    """
    Function to calculate the vapour pressure deficit
    Args:
        svp (float): saturated vapour pressure (kPa)
        rh (float): relative humidity (percent)
        
    Returns:
        float: vapour pressure deficit (kPa)
    """
    return svp * (1 - (rh / 100))