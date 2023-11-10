"""Methods for calculating the Solar Reflective Index (SRI) of materials and constructions."""

import numpy as np
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.opaque import EnergyMaterial
from ..bhom import decorator_factory


@decorator_factory()
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
            "Solar reflectance must be between 0 (black body) and 1 (white-body)."
        )
    if not 0 <= thermal_emittance <= 1:
        raise ValueError(
            "Thermal emittance must be between 0 (white-body) and 1 (black-body)."
        )
    if wind_speed < 0:
        raise ValueError("Wind speed must be greater than 0.")

    # convert wind speed to wind convection coeffecient
    speeds = [1, 4, 8]
    coeffs = [5, 12, 30]
    wind_convection_coefficient = np.interp(wind_speed, speeds, coeffs)

    # set the sensitivity threshold for the iterative calculation. Lower is more accurate but slower.
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
                f"SRI calculation did not converge. Surface temperature of {surface_temperature - 273.15}C."
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


@decorator_factory()
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


@decorator_factory()
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
