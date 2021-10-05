/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

using System.ComponentModel;

namespace LadybugTools_oM.Enums
{
    public enum EnergyPlusResultType
    {
        Undefined,
        [Description("Air System Component Model Simulation Calls")]
        AirSystemComponentModelSimulationCalls,
        [Description("Air System Relief Air Total Heat Loss Energy")]
        AirSystemReliefAirTotalHeatLossEnergy,
        [Description("Air System Simulation Iteration Count")]
        AirSystemSimulationIterationCount,
        [Description("Air System Simulation Maximum Iteration Count")]
        AirSystemSimulationMaximumIterationCount,
        [Description("Air System Solver Iteration Count")]
        AirSystemSolverIterationCount,
        [Description("Debug Surface Solar Shading Model DifShdgRatioHoriz")]
        DebugSurfaceSolarShadingModelDifShdgRatioHoriz,
        [Description("Debug Surface Solar Shading Model DifShdgRatioIsoSky")]
        DebugSurfaceSolarShadingModelDifShdgRatioIsoSky,
        [Description("Debug Surface Solar Shading Model WithShdgIsoSky")]
        DebugSurfaceSolarShadingModelWithShdgIsoSky,
        [Description("Debug Surface Solar Shading Model WoShdgIsoSky")]
        DebugSurfaceSolarShadingModelWoShdgIsoSky,
        [Description("Environmental Impact Total CH4 Emissions Carbon Equivalent Mass")]
        EnvironmentalImpactTotalCH4EmissionsCarbonEquivalentMass,
        [Description("Environmental Impact Total CO2 Emissions Carbon Equivalent Mass")]
        EnvironmentalImpactTotalCO2EmissionsCarbonEquivalentMass,
        [Description("Environmental Impact Total N2O Emissions Carbon Equivalent Mass")]
        EnvironmentalImpactTotalN2OEmissionsCarbonEquivalentMass,
        [Description("Facility All Zones Ventilation At Target Voz Time")]
        FacilityAllZonesVentilationAtTargetVozTime,
        [Description("Facility Any Zone Oscillating Temperatures During Occupancy Time")]
        FacilityAnyZoneOscillatingTemperaturesDuringOccupancyTime,
        [Description("Facility Any Zone Oscillating Temperatures Time")]
        FacilityAnyZoneOscillatingTemperaturesTime,
        [Description("Facility Any Zone Oscillating Temperatures in Deadband Time")]
        FacilityAnyZoneOscillatingTemperaturesinDeadbandTime,
        [Description("Facility Any Zone Ventilation Above Target Voz Time")]
        FacilityAnyZoneVentilationAboveTargetVozTime,
        [Description("Facility Any Zone Ventilation Below Target Voz Time")]
        FacilityAnyZoneVentilationBelowTargetVozTime,
        [Description("Facility Any Zone Ventilation When Unoccupied Time")]
        FacilityAnyZoneVentilationWhenUnoccupiedTime,
        [Description("Facility Cooling Setpoint Not Met Time")]
        FacilityCoolingSetpointNotMetTime,
        [Description("Facility Cooling Setpoint Not Met While Occupied Time")]
        FacilityCoolingSetpointNotMetWhileOccupiedTime,
        [Description("Facility Heating Setpoint Not Met Time")]
        FacilityHeatingSetpointNotMetTime,
        [Description("Facility Heating Setpoint Not Met While Occupied Time")]
        FacilityHeatingSetpointNotMetWhileOccupiedTime,
        [Description("Facility Thermal Comfort ASHRAE 55 Simple Model Summer Clothes Not Comfortable Time")]
        FacilityThermalComfortASHRAE55SimpleModelSummerClothesNotComfortableTime,
        [Description("Facility Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time")]
        FacilityThermalComfortASHRAE55SimpleModelSummerorWinterClothesNotComfortableTime,
        [Description("Facility Thermal Comfort ASHRAE 55 Simple Model Winter Clothes Not Comfortable Time")]
        FacilityThermalComfortASHRAE55SimpleModelWinterClothesNotComfortableTime,
        [Description("HVAC System Solver Iteration Count")]
        HVACSystemSolverIterationCount,
        [Description("HVAC System Total Heat Rejection Energy")]
        HVACSystemTotalHeatRejectionEnergy,
        [Description("Schedule Value")]
        ScheduleValue,
        [Description("Site Beam Solar Radiation Luminous Efficacy")]
        SiteBeamSolarRadiationLuminousEfficacy,
        [Description("Site Day Type Index")]
        SiteDayTypeIndex,
        [Description("Site Daylight Saving Time Status")]
        SiteDaylightSavingTimeStatus,
        [Description("Site Daylighting Model Sky Brightness")]
        SiteDaylightingModelSkyBrightness,
        [Description("Site Daylighting Model Sky Clearness")]
        SiteDaylightingModelSkyClearness,
        [Description("Site Deep Ground Temperature")]
        SiteDeepGroundTemperature,
        [Description("Site Diffuse Solar Radiation Rate per Area")]
        SiteDiffuseSolarRadiationRateperArea,
        [Description("Site Direct Solar Radiation Rate per Area")]
        SiteDirectSolarRadiationRateperArea,
        [Description("Site Exterior Beam Normal Illuminance")]
        SiteExteriorBeamNormalIlluminance,
        [Description("Site Exterior Horizontal Beam Illuminance")]
        SiteExteriorHorizontalBeamIlluminance,
        [Description("Site Exterior Horizontal Sky Illuminance")]
        SiteExteriorHorizontalSkyIlluminance,
        [Description("Site Ground Reflected Solar Radiation Rate per Area")]
        SiteGroundReflectedSolarRadiationRateperArea,
        [Description("Site Ground Temperature")]
        SiteGroundTemperature,
        [Description("Site Horizontal Infrared Radiation Rate per Area")]
        SiteHorizontalInfraredRadiationRateperArea,
        [Description("Site Mains Water Temperature")]
        SiteMainsWaterTemperature,
        [Description("Site Opaque Sky Cover")]
        SiteOpaqueSkyCover,
        [Description("Site Outdoor Air Barometric Pressure")]
        SiteOutdoorAirBarometricPressure,
        [Description("Site Outdoor Air Density")]
        SiteOutdoorAirDensity,
        [Description("Site Outdoor Air Dewpoint Temperature")]
        SiteOutdoorAirDewpointTemperature,
        [Description("Site Outdoor Air Drybulb Temperature")]
        SiteOutdoorAirDrybulbTemperature,
        [Description("Site Outdoor Air Enthalpy")]
        SiteOutdoorAirEnthalpy,
        [Description("Site Outdoor Air Humidity Ratio")]
        SiteOutdoorAirHumidityRatio,
        [Description("Site Outdoor Air Relative Humidity")]
        SiteOutdoorAirRelativeHumidity,
        [Description("Site Outdoor Air Wetbulb Temperature")]
        SiteOutdoorAirWetbulbTemperature,
        [Description("Site Precipitation Depth")]
        SitePrecipitationDepth,
        [Description("Site Rain Status")]
        SiteRainStatus,
        [Description("Site Simple Factor Model Ground Temperature")]
        SiteSimpleFactorModelGroundTemperature,
        [Description("Site Sky Diffuse Solar Radiation Luminous Efficacy")]
        SiteSkyDiffuseSolarRadiationLuminousEfficacy,
        [Description("Site Sky Temperature")]
        SiteSkyTemperature,
        [Description("Site Snow on Ground Status")]
        SiteSnowonGroundStatus,
        [Description("Site Solar Altitude Angle")]
        SiteSolarAltitudeAngle,
        [Description("Site Solar Azimuth Angle")]
        SiteSolarAzimuthAngle,
        [Description("Site Solar Hour Angle")]
        SiteSolarHourAngle,
        [Description("Site Surface Ground Temperature")]
        SiteSurfaceGroundTemperature,
        [Description("Site Total Sky Cover")]
        SiteTotalSkyCover,
        [Description("Site Total Surface Heat Emission to Air")]
        SiteTotalSurfaceHeatEmissiontoAir,
        [Description("Site Total Zone Exfiltration Heat Loss")]
        SiteTotalZoneExfiltrationHeatLoss,
        [Description("Site Total Zone Exhaust Air Heat Loss")]
        SiteTotalZoneExhaustAirHeatLoss,
        [Description("Site Wind Direction")]
        SiteWindDirection,
        [Description("Site Wind Speed")]
        SiteWindSpeed,
        [Description("Surface Anisotropic Sky Multiplier")]
        SurfaceAnisotropicSkyMultiplier,
        [Description("Surface Average Face Conduction Heat Gain Rate")]
        SurfaceAverageFaceConductionHeatGainRate,
        [Description("Surface Average Face Conduction Heat Loss Rate")]
        SurfaceAverageFaceConductionHeatLossRate,
        [Description("Surface Average Face Conduction Heat Transfer Energy")]
        SurfaceAverageFaceConductionHeatTransferEnergy,
        [Description("Surface Average Face Conduction Heat Transfer Rate per Area")]
        SurfaceAverageFaceConductionHeatTransferRateperArea,
        [Description("Surface Average Face Conduction Heat Transfer Rate")]
        SurfaceAverageFaceConductionHeatTransferRate,
        [Description("Surface Heat Storage Energy")]
        SurfaceHeatStorageEnergy,
        [Description("Surface Heat Storage Gain Rate")]
        SurfaceHeatStorageGainRate,
        [Description("Surface Heat Storage Loss Rate")]
        SurfaceHeatStorageLossRate,
        [Description("Surface Heat Storage Rate per Area")]
        SurfaceHeatStorageRateperArea,
        [Description("Surface Heat Storage Rate")]
        SurfaceHeatStorageRate,
        [Description("Surface Inside Face Absorbed Shortwave Radiation Rate")]
        SurfaceInsideFaceAbsorbedShortwaveRadiationRate,
        [Description("Surface Inside Face Adjacent Air Temperature")]
        SurfaceInsideFaceAdjacentAirTemperature,
        [Description("Surface Inside Face Beam Solar Radiation Heat Gain Rate")]
        SurfaceInsideFaceBeamSolarRadiationHeatGainRate,
        [Description("Surface Inside Face Conduction Heat Gain Rate")]
        SurfaceInsideFaceConductionHeatGainRate,
        [Description("Surface Inside Face Conduction Heat Loss Rate")]
        SurfaceInsideFaceConductionHeatLossRate,
        [Description("Surface Inside Face Conduction Heat Transfer Energy")]
        SurfaceInsideFaceConductionHeatTransferEnergy,
        [Description("Surface Inside Face Conduction Heat Transfer Rate per Area")]
        SurfaceInsideFaceConductionHeatTransferRateperArea,
        [Description("Surface Inside Face Conduction Heat Transfer Rate")]
        SurfaceInsideFaceConductionHeatTransferRate,
        [Description("Surface Inside Face Convection Classification Index")]
        SurfaceInsideFaceConvectionClassificationIndex,
        [Description("Surface Inside Face Convection Heat Gain Energy")]
        SurfaceInsideFaceConvectionHeatGainEnergy,
        [Description("Surface Inside Face Convection Heat Gain Rate per Area")]
        SurfaceInsideFaceConvectionHeatGainRateperArea,
        [Description("Surface Inside Face Convection Heat Gain Rate")]
        SurfaceInsideFaceConvectionHeatGainRate,
        [Description("Surface Inside Face Convection Heat Transfer Coefficient")]
        SurfaceInsideFaceConvectionHeatTransferCoefficient,
        [Description("Surface Inside Face Convection Model Equation Index")]
        SurfaceInsideFaceConvectionModelEquationIndex,
        [Description("Surface Inside Face Convection Reference Air Index")]
        SurfaceInsideFaceConvectionReferenceAirIndex,
        [Description("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Energy")]
        SurfaceInsideFaceExteriorWindowsIncidentBeamSolarRadiationEnergy,
        [Description("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate per Area")]
        SurfaceInsideFaceExteriorWindowsIncidentBeamSolarRadiationRateperArea,
        [Description("Surface Inside Face Exterior Windows Incident Beam Solar Radiation Rate")]
        SurfaceInsideFaceExteriorWindowsIncidentBeamSolarRadiationRate,
        [Description("Surface Inside Face Heat Source Gain Rate per Area")]
        SurfaceInsideFaceHeatSourceGainRateperArea,
        [Description("Surface Inside Face Initial Transmitted Diffuse Absorbed Solar Radiation Rate")]
        SurfaceInsideFaceInitialTransmittedDiffuseAbsorbedSolarRadiationRate,
        [Description("Surface Inside Face Initial Transmitted Diffuse Transmitted Out Window Solar Radiation Rate")]
        SurfaceInsideFaceInitialTransmittedDiffuseTransmittedOutWindowSolarRadiationRate,
        [Description("Surface Inside Face Interior Movable Insulation Temperature")]
        SurfaceInsideFaceInteriorMovableInsulationTemperature,
        [Description("Surface Inside Face Interior Windows Incident Beam Solar Radiation Energy")]
        SurfaceInsideFaceInteriorWindowsIncidentBeamSolarRadiationEnergy,
        [Description("Surface Inside Face Interior Windows Incident Beam Solar Radiation Rate per Area")]
        SurfaceInsideFaceInteriorWindowsIncidentBeamSolarRadiationRateperArea,
        [Description("Surface Inside Face Interior Windows Incident Beam Solar Radiation Rate")]
        SurfaceInsideFaceInteriorWindowsIncidentBeamSolarRadiationRate,
        [Description("Surface Inside Face Internal Gains Radiation Heat Gain Energy")]
        SurfaceInsideFaceInternalGainsRadiationHeatGainEnergy,
        [Description("Surface Inside Face Internal Gains Radiation Heat Gain Rate per Area")]
        SurfaceInsideFaceInternalGainsRadiationHeatGainRateperArea,
        [Description("Surface Inside Face Internal Gains Radiation Heat Gain Rate")]
        SurfaceInsideFaceInternalGainsRadiationHeatGainRate,
        [Description("Surface Inside Face Lights Radiation Heat Gain Energy")]
        SurfaceInsideFaceLightsRadiationHeatGainEnergy,
        [Description("Surface Inside Face Lights Radiation Heat Gain Rate per Area")]
        SurfaceInsideFaceLightsRadiationHeatGainRateperArea,
        [Description("Surface Inside Face Lights Radiation Heat Gain Rate")]
        SurfaceInsideFaceLightsRadiationHeatGainRate,
        [Description("Surface Inside Face Net Surface Thermal Radiation Heat Gain Energy")]
        SurfaceInsideFaceNetSurfaceThermalRadiationHeatGainEnergy,
        [Description("Surface Inside Face Net Surface Thermal Radiation Heat Gain Rate per Area")]
        SurfaceInsideFaceNetSurfaceThermalRadiationHeatGainRateperArea,
        [Description("Surface Inside Face Net Surface Thermal Radiation Heat Gain Rate")]
        SurfaceInsideFaceNetSurfaceThermalRadiationHeatGainRate,
        [Description("Surface Inside Face Solar Radiation Heat Gain Energy")]
        SurfaceInsideFaceSolarRadiationHeatGainEnergy,
        [Description("Surface Inside Face Solar Radiation Heat Gain Rate per Area")]
        SurfaceInsideFaceSolarRadiationHeatGainRateperArea,
        [Description("Surface Inside Face Solar Radiation Heat Gain Rate")]
        SurfaceInsideFaceSolarRadiationHeatGainRate,
        [Description("Surface Inside Face System Radiation Heat Gain Energy")]
        SurfaceInsideFaceSystemRadiationHeatGainEnergy,
        [Description("Surface Inside Face System Radiation Heat Gain Rate per Area")]
        SurfaceInsideFaceSystemRadiationHeatGainRateperArea,
        [Description("Surface Inside Face System Radiation Heat Gain Rate")]
        SurfaceInsideFaceSystemRadiationHeatGainRate,
        [Description("Surface Inside Face Temperature")]
        SurfaceInsideFaceTemperature,
        [Description("Surface Outside Face Beam Solar Incident Angle Cosine Value")]
        SurfaceOutsideFaceBeamSolarIncidentAngleCosineValue,
        [Description("Surface Outside Face Conduction Heat Gain Rate")]
        SurfaceOutsideFaceConductionHeatGainRate,
        [Description("Surface Outside Face Conduction Heat Loss Rate")]
        SurfaceOutsideFaceConductionHeatLossRate,
        [Description("Surface Outside Face Conduction Heat Transfer Energy")]
        SurfaceOutsideFaceConductionHeatTransferEnergy,
        [Description("Surface Outside Face Conduction Heat Transfer Rate per Area")]
        SurfaceOutsideFaceConductionHeatTransferRateperArea,
        [Description("Surface Outside Face Conduction Heat Transfer Rate")]
        SurfaceOutsideFaceConductionHeatTransferRate,
        [Description("Surface Outside Face Convection Classification Index")]
        SurfaceOutsideFaceConvectionClassificationIndex,
        [Description("Surface Outside Face Convection Heat Gain Energy")]
        SurfaceOutsideFaceConvectionHeatGainEnergy,
        [Description("Surface Outside Face Convection Heat Gain Rate per Area")]
        SurfaceOutsideFaceConvectionHeatGainRateperArea,
        [Description("Surface Outside Face Convection Heat Gain Rate")]
        SurfaceOutsideFaceConvectionHeatGainRate,
        [Description("Surface Outside Face Convection Heat Transfer Coefficient")]
        SurfaceOutsideFaceConvectionHeatTransferCoefficient,
        [Description("Surface Outside Face Forced Convection Model Equation Index")]
        SurfaceOutsideFaceForcedConvectionModelEquationIndex,
        [Description("Surface Outside Face Heat Emission to Air Rate")]
        SurfaceOutsideFaceHeatEmissiontoAirRate,
        [Description("Surface Outside Face Heat Source Gain Rate per Area")]
        SurfaceOutsideFaceHeatSourceGainRateperArea,
        [Description("Surface Outside Face Incident Beam Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentBeamSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Beam To Beam Surface Reflected Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentBeamToBeamSurfaceReflectedSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Beam To Diffuse Ground Reflected Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentBeamToDiffuseGroundReflectedSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Beam To Diffuse Surface Reflected Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentBeamToDiffuseSurfaceReflectedSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Ground Diffuse Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentGroundDiffuseSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Sky Diffuse Ground Reflected Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentSkyDiffuseGroundReflectedSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentSkyDiffuseSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Sky Diffuse Surface Reflected Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentSkyDiffuseSurfaceReflectedSolarRadiationRateperArea,
        [Description("Surface Outside Face Incident Solar Radiation Rate per Area")]
        SurfaceOutsideFaceIncidentSolarRadiationRateperArea,
        [Description("Surface Outside Face Natural Convection Model Equation Index")]
        SurfaceOutsideFaceNaturalConvectionModelEquationIndex,
        [Description("Surface Outside Face Net Thermal Radiation Heat Gain Energy")]
        SurfaceOutsideFaceNetThermalRadiationHeatGainEnergy,
        [Description("Surface Outside Face Net Thermal Radiation Heat Gain Rate per Area")]
        SurfaceOutsideFaceNetThermalRadiationHeatGainRateperArea,
        [Description("Surface Outside Face Net Thermal Radiation Heat Gain Rate")]
        SurfaceOutsideFaceNetThermalRadiationHeatGainRate,
        [Description("Surface Outside Face Outdoor Air Drybulb Temperature")]
        SurfaceOutsideFaceOutdoorAirDrybulbTemperature,
        [Description("Surface Outside Face Outdoor Air Wetbulb Temperature")]
        SurfaceOutsideFaceOutdoorAirWetbulbTemperature,
        [Description("Surface Outside Face Outdoor Air Wind Direction")]
        SurfaceOutsideFaceOutdoorAirWindDirection,
        [Description("Surface Outside Face Outdoor Air Wind Speed")]
        SurfaceOutsideFaceOutdoorAirWindSpeed,
        [Description("Surface Outside Face Solar Radiation Heat Gain Energy")]
        SurfaceOutsideFaceSolarRadiationHeatGainEnergy,
        [Description("Surface Outside Face Solar Radiation Heat Gain Rate per Area")]
        SurfaceOutsideFaceSolarRadiationHeatGainRateperArea,
        [Description("Surface Outside Face Solar Radiation Heat Gain Rate")]
        SurfaceOutsideFaceSolarRadiationHeatGainRate,
        [Description("Surface Outside Face Sunlit Area")]
        SurfaceOutsideFaceSunlitArea,
        [Description("Surface Outside Face Sunlit Fraction")]
        SurfaceOutsideFaceSunlitFraction,
        [Description("Surface Outside Face Temperature")]
        SurfaceOutsideFaceTemperature,
        [Description("Surface Outside Face Thermal Radiation to Air Heat Transfer Coefficient")]
        SurfaceOutsideFaceThermalRadiationtoAirHeatTransferCoefficient,
        [Description("Surface Outside Face Thermal Radiation to Air Heat Transfer Rate")]
        SurfaceOutsideFaceThermalRadiationtoAirHeatTransferRate,
        [Description("Surface Outside Face Thermal Radiation to Ground Heat Transfer Coefficient")]
        SurfaceOutsideFaceThermalRadiationtoGroundHeatTransferCoefficient,
        [Description("Surface Outside Face Thermal Radiation to Sky Heat Transfer Coefficient")]
        SurfaceOutsideFaceThermalRadiationtoSkyHeatTransferCoefficient,
        [Description("Surface Outside Normal Azimuth Angle")]
        SurfaceOutsideNormalAzimuthAngle,
        [Description("Surface Window BSDF Beam Direction Number")]
        SurfaceWindowBSDFBeamDirectionNumber,
        [Description("Surface Window BSDF Beam Phi Angle")]
        SurfaceWindowBSDFBeamPhiAngle,
        [Description("Surface Window BSDF Beam Theta Angle")]
        SurfaceWindowBSDFBeamThetaAngle,
        [Description("System Node Current Density Volume Flow Rate")]
        SystemNodeCurrentDensityVolumeFlowRate,
        [Description("System Node Current Density")]
        SystemNodeCurrentDensity,
        [Description("System Node Dewpoint Temperature")]
        SystemNodeDewpointTemperature,
        [Description("System Node Enthalpy")]
        SystemNodeEnthalpy,
        [Description("System Node Height")]
        SystemNodeHeight,
        [Description("System Node Humidity Ratio")]
        SystemNodeHumidityRatio,
        [Description("System Node Mass Flow Rate")]
        SystemNodeMassFlowRate,
        [Description("System Node Pressure")]
        SystemNodePressure,
        [Description("System Node Quality")]
        SystemNodeQuality,
        [Description("System Node Relative Humidity")]
        SystemNodeRelativeHumidity,
        [Description("System Node Setpoint High Temperature")]
        SystemNodeSetpointHighTemperature,
        [Description("System Node Setpoint Humidity Ratio")]
        SystemNodeSetpointHumidityRatio,
        [Description("System Node Setpoint Low Temperature")]
        SystemNodeSetpointLowTemperature,
        [Description("System Node Setpoint Maximum Humidity Ratio")]
        SystemNodeSetpointMaximumHumidityRatio,
        [Description("System Node Setpoint Minimum Humidity Ratio")]
        SystemNodeSetpointMinimumHumidityRatio,
        [Description("System Node Setpoint Temperature")]
        SystemNodeSetpointTemperature,
        [Description("System Node Specific Heat")]
        SystemNodeSpecificHeat,
        [Description("System Node Standard Density Volume Flow Rate")]
        SystemNodeStandardDensityVolumeFlowRate,
        [Description("System Node Temperature")]
        SystemNodeTemperature,
        [Description("System Node Wetbulb Temperature")]
        SystemNodeWetbulbTemperature,
        [Description("System Node Wind Direction")]
        SystemNodeWindDirection,
        [Description("System Node Wind Speed")]
        SystemNodeWindSpeed,
        [Description("Zone Adaptive Comfort Operative Temperature Set Point")]
        ZoneAdaptiveComfortOperativeTemperatureSetPoint,
        [Description("Zone Air Heat Balance Air Energy Storage Rate")]
        ZoneAirHeatBalanceAirEnergyStorageRate,
        [Description("Zone Air Heat Balance Internal Convective Heat Gain Rate")]
        ZoneAirHeatBalanceInternalConvectiveHeatGainRate,
        [Description("Zone Air Heat Balance Interzone Air Transfer Rate")]
        ZoneAirHeatBalanceInterzoneAirTransferRate,
        [Description("Zone Air Heat Balance Outdoor Air Transfer Rate")]
        ZoneAirHeatBalanceOutdoorAirTransferRate,
        [Description("Zone Air Heat Balance Surface Convection Rate")]
        ZoneAirHeatBalanceSurfaceConvectionRate,
        [Description("Zone Air Heat Balance System Air Transfer Rate")]
        ZoneAirHeatBalanceSystemAirTransferRate,
        [Description("Zone Air Heat Balance System Convective Heat Gain Rate")]
        ZoneAirHeatBalanceSystemConvectiveHeatGainRate,
        [Description("Zone Air Humidity Ratio")]
        ZoneAirHumidityRatio,
        [Description("Zone Air Relative Humidity")]
        ZoneAirRelativeHumidity,
        [Description("Zone Air System Sensible Cooling Energy")]
        ZoneAirSystemSensibleCoolingEnergy,
        [Description("Zone Air System Sensible Cooling Rate")]
        ZoneAirSystemSensibleCoolingRate,
        [Description("Zone Air System Sensible Heating Energy")]
        ZoneAirSystemSensibleHeatingEnergy,
        [Description("Zone Air System Sensible Heating Rate")]
        ZoneAirSystemSensibleHeatingRate,
        [Description("Zone Air Temperature")]
        ZoneAirTemperature,
        [Description("Zone Cooling Setpoint Not Met Time")]
        ZoneCoolingSetpointNotMetTime,
        [Description("Zone Cooling Setpoint Not Met While Occupied Time")]
        ZoneCoolingSetpointNotMetWhileOccupiedTime,
        [Description("Zone Exfiltration Heat Transfer Rate")]
        ZoneExfiltrationHeatTransferRate,
        [Description("Zone Exfiltration Latent Heat Transfer Rate")]
        ZoneExfiltrationLatentHeatTransferRate,
        [Description("Zone Exfiltration Sensible Heat Transfer Rate")]
        ZoneExfiltrationSensibleHeatTransferRate,
        [Description("Zone Exhaust Air Heat Transfer Rate")]
        ZoneExhaustAirHeatTransferRate,
        [Description("Zone Exhaust Air Latent Heat Transfer Rate")]
        ZoneExhaustAirLatentHeatTransferRate,
        [Description("Zone Exhaust Air Sensible Heat Transfer Rate")]
        ZoneExhaustAirSensibleHeatTransferRate,
        [Description("Zone Exterior Windows Total Transmitted Beam Solar Radiation Energy")]
        ZoneExteriorWindowsTotalTransmittedBeamSolarRadiationEnergy,
        [Description("Zone Exterior Windows Total Transmitted Beam Solar Radiation Rate")]
        ZoneExteriorWindowsTotalTransmittedBeamSolarRadiationRate,
        [Description("Zone Exterior Windows Total Transmitted Diffuse Solar Radiation Energy")]
        ZoneExteriorWindowsTotalTransmittedDiffuseSolarRadiationEnergy,
        [Description("Zone Exterior Windows Total Transmitted Diffuse Solar Radiation Rate")]
        ZoneExteriorWindowsTotalTransmittedDiffuseSolarRadiationRate,
        [Description("Zone Heat Index")]
        ZoneHeatIndex,
        [Description("Zone Heating Setpoint Not Met Time")]
        ZoneHeatingSetpointNotMetTime,
        [Description("Zone Heating Setpoint Not Met While Occupied Time")]
        ZoneHeatingSetpointNotMetWhileOccupiedTime,
        [Description("Zone Humidity Index")]
        ZoneHumidityIndex,
        [Description("Zone Interior Windows Total Transmitted Beam Solar Radiation Energy")]
        ZoneInteriorWindowsTotalTransmittedBeamSolarRadiationEnergy,
        [Description("Zone Interior Windows Total Transmitted Beam Solar Radiation Rate")]
        ZoneInteriorWindowsTotalTransmittedBeamSolarRadiationRate,
        [Description("Zone Interior Windows Total Transmitted Diffuse Solar Radiation Energy")]
        ZoneInteriorWindowsTotalTransmittedDiffuseSolarRadiationEnergy,
        [Description("Zone Interior Windows Total Transmitted Diffuse Solar Radiation Rate")]
        ZoneInteriorWindowsTotalTransmittedDiffuseSolarRadiationRate,
        [Description("Zone Mean Air Dewpoint Temperature")]
        ZoneMeanAirDewpointTemperature,
        [Description("Zone Mean Air Humidity Ratio")]
        ZoneMeanAirHumidityRatio,
        [Description("Zone Mean Air Temperature")]
        ZoneMeanAirTemperature,
        [Description("Zone Mean Radiant Temperature")]
        ZoneMeanRadiantTemperature,
        [Description("Zone Operative Temperature")]
        ZoneOperativeTemperature,
        [Description("Zone Oscillating Temperatures During Occupancy Time")]
        ZoneOscillatingTemperaturesDuringOccupancyTime,
        [Description("Zone Oscillating Temperatures Time")]
        ZoneOscillatingTemperaturesTime,
        [Description("Zone Oscillating Temperatures in Deadband Time")]
        ZoneOscillatingTemperaturesinDeadbandTime,
        [Description("Zone Outdoor Air Drybulb Temperature")]
        ZoneOutdoorAirDrybulbTemperature,
        [Description("Zone Outdoor Air Wetbulb Temperature")]
        ZoneOutdoorAirWetbulbTemperature,
        [Description("Zone Outdoor Air Wind Direction")]
        ZoneOutdoorAirWindDirection,
        [Description("Zone Outdoor Air Wind Speed")]
        ZoneOutdoorAirWindSpeed,
        [Description("Zone Predicted Moisture Load Moisture Transfer Rate")]
        ZonePredictedMoistureLoadMoistureTransferRate,
        [Description("Zone Predicted Moisture Load to Dehumidifying Setpoint Moisture Transfer Rate")]
        ZonePredictedMoistureLoadtoDehumidifyingSetpointMoistureTransferRate,
        [Description("Zone Predicted Moisture Load to Humidifying Setpoint Moisture Transfer Rate")]
        ZonePredictedMoistureLoadtoHumidifyingSetpointMoistureTransferRate,
        [Description("Zone Predicted Sensible Load Room Air Correction Factor")]
        ZonePredictedSensibleLoadRoomAirCorrectionFactor,
        [Description("Zone Predicted Sensible Load to Cooling Setpoint Heat Transfer Rate")]
        ZonePredictedSensibleLoadtoCoolingSetpointHeatTransferRate,
        [Description("Zone Predicted Sensible Load to Heating Setpoint Heat Transfer Rate")]
        ZonePredictedSensibleLoadtoHeatingSetpointHeatTransferRate,
        [Description("Zone Predicted Sensible Load to Setpoint Heat Transfer Rate")]
        ZonePredictedSensibleLoadtoSetpointHeatTransferRate,
        [Description("Zone System Predicted Moisture Load Moisture Transfer Rate")]
        ZoneSystemPredictedMoistureLoadMoistureTransferRate,
        [Description("Zone System Predicted Moisture Load to Dehumidifying Setpoint Moisture Transfer Rate")]
        ZoneSystemPredictedMoistureLoadtoDehumidifyingSetpointMoistureTransferRate,
        [Description("Zone System Predicted Moisture Load to Humidifying Setpoint Moisture Transfer Rate")]
        ZoneSystemPredictedMoistureLoadtoHumidifyingSetpointMoistureTransferRate,
        [Description("Zone System Predicted Sensible Load to Cooling Setpoint Heat Transfer Rate")]
        ZoneSystemPredictedSensibleLoadtoCoolingSetpointHeatTransferRate,
        [Description("Zone System Predicted Sensible Load to Heating Setpoint Heat Transfer Rate")]
        ZoneSystemPredictedSensibleLoadtoHeatingSetpointHeatTransferRate,
        [Description("Zone System Predicted Sensible Load to Setpoint Heat Transfer Rate")]
        ZoneSystemPredictedSensibleLoadtoSetpointHeatTransferRate,
        [Description("Zone Thermal Comfort ASHRAE 55 Simple Model Summer Clothes Not Comfortable Time")]
        ZoneThermalComfortASHRAE55SimpleModelSummerClothesNotComfortableTime,
        [Description("Zone Thermal Comfort ASHRAE 55 Simple Model Summer or Winter Clothes Not Comfortable Time")]
        ZoneThermalComfortASHRAE55SimpleModelSummerorWinterClothesNotComfortableTime,
        [Description("Zone Thermal Comfort ASHRAE 55 Simple Model Winter Clothes Not Comfortable Time")]
        ZoneThermalComfortASHRAE55SimpleModelWinterClothesNotComfortableTime,
        [Description("Zone Thermostat Air Temperature")]
        ZoneThermostatAirTemperature,
        [Description("Zone Thermostat Control Type")]
        ZoneThermostatControlType,
        [Description("Zone Thermostat Cooling Setpoint Temperature")]
        ZoneThermostatCoolingSetpointTemperature,
        [Description("Zone Thermostat Heating Setpoint Temperature")]
        ZoneThermostatHeatingSetpointTemperature,
        [Description("Zone Total Internal Convective Heating Energy")]
        ZoneTotalInternalConvectiveHeatingEnergy,
        [Description("Zone Total Internal Convective Heating Rate")]
        ZoneTotalInternalConvectiveHeatingRate,
        [Description("Zone Total Internal Latent Gain Energy")]
        ZoneTotalInternalLatentGainEnergy,
        [Description("Zone Total Internal Latent Gain Rate")]
        ZoneTotalInternalLatentGainRate,
        [Description("Zone Total Internal Radiant Heating Energy")]
        ZoneTotalInternalRadiantHeatingEnergy,
        [Description("Zone Total Internal Radiant Heating Rate")]
        ZoneTotalInternalRadiantHeatingRate,
        [Description("Zone Total Internal Total Heating Energy")]
        ZoneTotalInternalTotalHeatingEnergy,
        [Description("Zone Total Internal Total Heating Rate")]
        ZoneTotalInternalTotalHeatingRate,
        [Description("Zone Total Internal Visible Radiation Heating Energy")]
        ZoneTotalInternalVisibleRadiationHeatingEnergy,
        [Description("Zone Total Internal Visible Radiation Heating Rate")]
        ZoneTotalInternalVisibleRadiationHeatingRate,
        [Description("Zone Windows Total Heat Gain Energy")]
        ZoneWindowsTotalHeatGainEnergy,
        [Description("Zone Windows Total Heat Gain Rate")]
        ZoneWindowsTotalHeatGainRate,
        [Description("Zone Windows Total Heat Loss Energy")]
        ZoneWindowsTotalHeatLossEnergy,
        [Description("Zone Windows Total Heat Loss Rate")]
        ZoneWindowsTotalHeatLossRate,
        [Description("Zone Windows Total Transmitted Solar Radiation Energy")]
        ZoneWindowsTotalTransmittedSolarRadiationEnergy,
        [Description("Zone Windows Total Transmitted Solar Radiation Rate")]
        ZoneWindowsTotalTransmittedSolarRadiationRate,
    }
}
