"""Methods for calculating the Solar Reflective Index (SRI) of materials and constructions."""

import numpy as np
import pandas as pd
from honeybee.typing import clean_string
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial
from honeybee_radiance.modifier.material import Plastic
from matplotlib.colors import hex2color
from sklearn.linear_model import LinearRegression

from .. import SRI_DATA
from ..bhom.analytics import bhom_analytics
from ..bhom.logging import CONSOLE_LOGGER


@bhom_analytics()
def calculate_sri(
    solar_reflectance: float,
    thermal_emittance: float,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a material from its reflectance and emittance.
    Note, this method assumes a horizontal material (facing the sky).

    This method is based on the tool created by Ronnen Levinson, Heat Island
    Group, LBNL. It uses the method from ASTM Standard E1980-11 to calculate
    the SRI of a material, given its solar reflectance and thermal emittance.

    Wind speed is used inestead of wind convection coeffeicnt, based on the
    suggested values in ASTM E1980-11.

    Args:
        solar_reflectance (float): Solar reflectance of the material. Unitless,
            between 0 (black body) and 1 (white-body).
        thermal_emittance (float): Thermal emittance of the material. Unitless,
            between 0 (white-body) and 1 (black-body).
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the material. Unitless.
    """

    if not 0 <= solar_reflectance <= 1:
        raise ValueError(
            "Solar reflectance must be between 0 (black body) - 1 (white-body)."
        )
    if not 0 <= thermal_emittance <= 1:
        raise ValueError(
            "Thermal emittance must be between 0 (white-body) - 1 (black-body)."
        )
    if wind_speed < 0:
        raise ValueError("Wind speed must be greater than 0.")

    # convert wind speed to wind convection coeffecient
    speeds = [1, 4, 8]
    coeffs = [5, 12, 30]
    wind_convection_coefficient = np.interp(wind_speed, speeds, coeffs)

    # set the sensitivity threshold for iterative calculation. Lower is more accurate but slower.
    threshold = 0.5  # W
    increment = 0.01  # K
    iterations = 100000

    air_temperature = air_temperature + 273.15  # K
    sky_temperature = sky_temperature + 273.15  # K

    sigma = 5.67e-8  # W m-2 K-4 Stefan-Boltzmann constant
    blackbody_solar_reflectance = 0.05  #  0 - 1
    whitebody_solar_reflectance = 0.8  #  0 - 1
    blackbody_thermal_emittance = 0.9  #  0 - 1
    whitebody_thermal_emittance = 0.9  #  0 - 1

    surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - solar_reflectance) * insolation
        - (
            thermal_emittance
            * sigma
            * (surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient * (surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError(
                ("SRI calculation did not converge. Surface temperature of "
                 f"{surface_temperature - 273.15}C.")
            )
        surface_temperature += increment

    blackbody_surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - blackbody_solar_reflectance) * insolation
        - (
            blackbody_thermal_emittance
            * sigma
            * (blackbody_surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient
            * (blackbody_surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError("SRI calculation did not converge.")
        blackbody_surface_temperature += increment

    whitebody_surface_temperature = 200.0  # K
    n = 0
    while not np.isclose(
        (1 - whitebody_solar_reflectance) * insolation
        - (
            whitebody_thermal_emittance
            * sigma
            * (whitebody_surface_temperature**4 - sky_temperature**4)
            + wind_convection_coefficient
            * (whitebody_surface_temperature - air_temperature)
        ),
        0,
        atol=threshold,
    ):
        n += 1
        if n > iterations:
            raise ValueError("SRI calculation did not converge.")
        whitebody_surface_temperature += increment

    solar_reflective_index = (
        100
        * (blackbody_surface_temperature - surface_temperature)
        / (blackbody_surface_temperature - whitebody_surface_temperature)
    )

    if solar_reflective_index < 0:
        return 0

    return solar_reflective_index


@bhom_analytics()
def material_sri(
    material: EnergyMaterial,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a Honeybee material.
    Note, this method assumes a horizontal material (facing the sky).

    Args:
        material (_EnergyMaterialOpaqueBase): A Honeybee opaque material.
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the material. Unitless.
    """

    return calculate_sri(
        solar_reflectance=material.solar_reflectance,
        thermal_emittance=material.thermal_absorptance,
        insolation=insolation,
        air_temperature=air_temperature,
        sky_temperature=sky_temperature,
        wind_speed=wind_speed,
    )


@bhom_analytics()
def construction_sri(
    construction: OpaqueConstruction,
    insolation: float = 1000,
    air_temperature: float = 36.85,
    sky_temperature: float = 26.85,
    wind_speed: float = 4,
) -> float:
    """Calculate the SRI of a Honeybee construction.
    Note, this method assumes a horizontal construction (facing the sky).

    Args:
        construction (Opaqueconstruction): A Honeybee construction material.
        insolation (float, optional): Insolation incident on the material.
            Defaults to 1000W/m2.
        air_temperature (float, optional): Air temperature. Defaults to 36.85C.
        sky_temperature (float, optional): Sky temperature. Defaults to 26.85C.
        wind_speed (float, optional): Speed of wind. Defaults to 4m/s.

    Returns:
        float: SRI of the construction. Unitless.
    """

    return calculate_sri(
        solar_reflectance=construction.outside_solar_reflectance,
        thermal_emittance=construction.outside_emissivity,
        insolation=insolation,
        air_temperature=air_temperature,
        sky_temperature=sky_temperature,
        wind_speed=wind_speed,
    )

@bhom_analytics()
def estimate_sri_from_color(
    color_hex: str,
    thermal_emittance: float = 0.9,
) -> float:
    """Estimate the SRI of a material from its color.

    Note:
    This uses predefined datasets from
     - https://www.deansteelbuildings.com/products/panels/paint-finish-color-selections/
     - https://www.berridge.com/resources/chart-of-sri-values/
    as the basis for the estimation of solar reflectance from colour luminosity.

    This method is an estimation and may not be accurate. It is usually 
    within ±10 of the actual SRI; however as full material properties are 
    not determinable from a HEX colour, then the resulting value must be 
    caveated with this warning! Metallic material finishes usually show 
    the largest discrepancy.

    Args:
        color_hex (str): 
            A color string in hex format.
        thermal_emittance (float): 
            Thermal emittance of the material. Unitless. Default is 0.9.

    Returns:
        float: 
            SRI of the material. Unitless.
    """

    # create a dataframe of the pre-defined colours and their respective SRI values
    df = pd.DataFrame(
        columns=["color", "reported_sr", "reported_sri", "hex"],
        data=[
            ["DSB Aluminum Zinc", 0.68, 55, "#dbdbe6"],
            ["DSB Oyster White", 0.52, 59, "#dfdbd4"],
            ["DSB Polar White", 0.61, 73, "#e2e8e7"],
            ["DSB Light Stone", 0.56, 65, "#cec5a8"],
            ["DSB Hawaiian Blue", 0.31, 31, "#406d84"],
            ["DSB Sahara Tan", 0.47, 53, "#ad9277"],
            ["DSB Ash Grey", 0.46, 52, "#a2a0a1"],
            ["DSB Burnished Bronze", 0.28, 29, "#372c26"],
            ["DSB Colony Green", 0.35, 37, "#657761"],
            ["DSB Fern Green", 0.29, 29, "#0b2d1d"],
            ["DSB Almond", 0.63, 75, "#ded8be"],
            ["DSB Snow White", 0.65, 78, "#e1e6e9"],
            ["DSB Brownstone", 0.47, 53, "#9f927f"],
            ["DSB Black", 0.25, 24, "#000000"],
            ["DSB Copper Metallic", 0.46, 51, "#b76e4c"],
            ["DSB Scarlet Red", 0.42, 47, "#d6141c"],
            ["DSB Harbor Blue", 0.26, 25, "#00405b"],
            ["DSB Hunter Green", 0.35, 39, "#1c453c"],
            ["DSB Roman Blue", 0.32, 33, "#366e87"],
            ["DSB Colonial Red", 0.34, 37, "#6b312d"],
            ["DSB Everglade", 0.33, 36, "#39706a"],
            ["DSB Slate Grey", 0.37, 41, "#82847f"],
            ["BER Aged Bronze", 0.31, 31, "#504533"],
            ["BER Almond", 0.65, 77, "#e1e5cc"],
            ["BER Bristol Blue", 0.33, 33, "#3a6c80"],
            ["BER Buckskin", 0.43, 46, "#847f67"],
            ["BER Burgundy", 0.32, 32, "#462523"],
            ["BER Charcoal Grey", 0.29, 28, "#444949"],
            ["BER Cityscape", 0.48, 54, "#99a1a7"],
            ["BER Colonial Red", 0.35, 35, "#6c2927"],
            ["BER Copper Brown", 0.32, 32, "#49382b"],
            ["BER Dark Bronze", 0.28, 27, "#3c362e"],
            ["BER Deep Red", 0.41, 44, "#a11e30"],
            ["BER Evergreen", 0.3, 29, "#325a43"],
            ["BER Forest Green", 0.3, 29, "#37614d"],
            ["BER Hartford Green", 0.27, 25, "#33463b"],
            ["BER Hemlock Green", 0.31, 31, "#597c6c"],
            ["BER Matte Black", 0.26, 24, "#282b2d"],
            ["BER Medium Bronze", 0.31, 31, "#5c4a36"],
            ["BER Parchment", 0.6, 71, "#d3d2c6"],
            ["BER Patina Green", 0.34, 35, "#5f9880"],
            ["BER Royal Blue", 0.27, 26, "#004862"],
            ["BER Shasta White", 0.61, 73, "#dde0dd"],
            ["BER Sierra Tan", 0.39, 42, "#a49a7a"],
            ["BER Teal Green", 0.26, 25, "#3d766f"],
            ["BER Terra - Cota", 0.36, 38, "#a44f38"],
            ["BER Zinc Grey", 0.39, 42, "#778082"],
            ["BER Acrylic-Coated Galvalume'", 0.67, 59, "#e6e8e8"],
            ["BER Award Blue", 0.17, 11, "#113f67"],
            ["BER Natural White", 0.71, 86, "#fffef8"],
            ["BER Antique Copper-Cote", 0.33, 34, "#818c6f"],
            ["BER Champagne", 0.4, 43, "#9b9484"],
            ["BER Copper-Cote", 0.51, 59, "#bc7038"],
            ["BER Lead-Cote", 0.36, 38, "#9c9999"],
            ["BER Preweathered Galvalume", 0.4, 43, "#7b8184"],
            ["BER Zinc-Cote", 0.53, 59, "#c9cbcc"],
            ["BER COR-TEN AZP Raw", 0.32, 34, "#a86a54"],
        ],
    )
    df.set_index("color", drop=True, inplace=True)
    df.sort_values("reported_sri", inplace=True)

    # add r, g, b as 0-1
    df["r"], df["g"], df["b"] = np.array([hex2color(i) for i in df.hex]).T

    # create radiance modifiers from the pre-defined colors, and determine solar reflectance
    df["radiance_material"] = [
        Plastic(
            identifier=clean_string(row),
            r_reflectance=i.r,
            g_reflectance=i.g,
            b_reflectance=i.b,
        )
        for row, i in df.iterrows()
    ]
    df["radiance_reflectivity"] = [i.average_reflectance for i in df.radiance_material]

    # get polynomial approximator for SR based on relationship between reported SR and Radiance calculated SR
    z = np.polyfit(df.dropna().radiance_reflectivity, df.dropna()["reported_sr"], 2)
    p = np.poly1d(z)

    # create dataframe for the passed colors
    color_r, color_g, color_b = hex2color(color_hex)
    color_plastic = Plastic(
        identifier="custom_color",
        r_reflectance=color_r,
        g_reflectance=color_g,
        b_reflectance=color_b,
    )
    color_estimated_sr = p(color_plastic.average_reflectance)
    color_estimated_sri = calculate_sri(
        solar_reflectance=color_estimated_sr,
        thermal_emittance=thermal_emittance,
    )

    return color_estimated_sri

@bhom_analytics()
def estimate_sri_properties(
    target_sri: float, target_emittance: float = 0.85, tolerance: float = 5
) -> tuple[float, float]:
    """Estimate solar absorptance and thermal emittance for a target SRI.

    This method uses a linear regression model to estimate the solar absorptance and
    thermal emittance of a material to achieve a target Solar Reflective Index (SRI).
    The model is trained on a dataset of these properties and the resultant SRI using
    fixed values for insolation, air_temperature, sky_temperature, and wind_speed.

    Args:
        target_sri (float): 
            The target Solar Reflective Index (SRI) for the material.
        target_emittance (float, optional): 
            The target Thermal Emittance for the material. Default is 0.85.
        tolerance (float, optional): 
            The acceptable tolerance between resultant SRI and target SRI. Default is 5.

    Returns:
        tuple[float, float]: The estimated solar absorptance and thermal emittance.
    """

    if target_sri < 0 or target_sri > 122:
        raise ValueError("Target SRI must be between 0 and 122.")

    if target_emittance <= 0 or target_emittance >= 1:
        raise ValueError(
            "Thermal absorptivity estimation is beyond allowable limits for the target SRI."
        )

    # load the dataset
    data = pd.read_csv(SRI_DATA, header=0)

    model = LinearRegression()
    model.fit(
        data[["solar_absorptivity", "thermal_absorptivity"]].values, data["sri"].values
    )

    possible_combinations = []
    sris = []
    for sa in np.linspace(0, 1, 101):
        sri = model.predict([[sa, target_emittance]])[0]
        if np.isclose(sri, target_sri, atol=tolerance):
            possible_combinations.append(sa)
            sris.append(sri)
    sa_ = round(np.mean(possible_combinations), 4)
    ta_ = round(target_emittance, 4)

    if sa_ <= 0 or sa_ >= 1:
        CONSOLE_LOGGER.error(
            (
                "Solar absorptivity estimation is beyond allowable limits "
                "for the target SRI."
            )
        )

    return float(sa_), ta_
