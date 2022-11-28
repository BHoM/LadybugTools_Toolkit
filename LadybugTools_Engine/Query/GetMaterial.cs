/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using Rhino.Geometry;
using Rhino.Render;
using System;
using System.ComponentModel;
using System.Linq.Expressions;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get a material object from it's Enum.")]
        [Output("material", "A material object to pass into the External Comfort workflow.")]
        public static ILBTMaterial GetMaterial(this Materials material)
        {
            if (material == Materials.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("A pre-defined material must be passed in order to return an object.");
            }
            
            switch (material)
            {
                case Materials.AsphaltPavement:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Asphalt Pavement",
                        Thickness = 0.2,
                        Conductivity = 0.75,
                        Density = 2360.0,
                        SpecificHeat = 920.0,
                        Roughness = Roughness.MediumRough,
                        ThermalAbsorptance = 0.93,
                        SolarAbsorptance = 0.87,
                        VisibleAbsorptance = 0.87,
                    };
                case Materials.ConcretePavement:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Concrete Pavement",
                        Thickness = 0.2,
                        Conductivity = 1.73,
                        Density = 2243.0,
                        SpecificHeat = 837.0,
                        Roughness = Roughness.MediumRough,
                        ThermalAbsorptance = 0.9,
                        SolarAbsorptance = 0.65,
                        VisibleAbsorptance = 0.65,
                    };

                case Materials.DryDust:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Dry Dust",
                        Thickness = 0.2,
                        Conductivity = 0.5,
                        Density = 1600.0,
                        SpecificHeat = 1026.0,
                        Roughness = Roughness.Rough,
                        ThermalAbsorptance = 0.9,
                        SolarAbsorptance = 0.7,
                        VisibleAbsorptance = 0.7,
                    };

                case Materials.DrySand:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Dry Sand",
                        Thickness = 0.2,
                        Conductivity = 0.33,
                        Density = 1555.0,
                        SpecificHeat = 800.0,
                        Roughness = Roughness.Rough,
                        ThermalAbsorptance = 0.85,
                        SolarAbsorptance = 0.65,
                        VisibleAbsorptance = 0.65,
                    };

                case Materials.GrassyLawn:
                    return new OpaqueVegetationMaterial()
                    {
                        Identifier = "Grassy Lawn",
                        Thickness = 0.1,
                        Conductivity = 0.35,
                        Density = 1100,
                        SpecificHeat = 1200,
                        Roughness = Roughness.MediumRough,
                        SoilThermalAbsorptance = 0.9,
                        SoilSolarAbsorptance = 0.7,
                        SoilVisibleAbsorptance = 0.7,
                        PlantHeight = 0.2,
                        LeafAreaIndex = 1.0,
                        LeafReflectivity = 0.22,
                        LeafEmissivity = 0.95,
                        MinStomatalResist = 180,
                    };

                case Materials.Metal:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Metal Surface",
                        Thickness = 0.0007999999999999979,
                        Conductivity = 45.24971874361766,
                        Density = 7824.017889489713,
                        SpecificHeat = 499.67760800730963,
                        Roughness = Roughness.Smooth,
                        ThermalAbsorptance = 0.9,
                        SolarAbsorptance = 0.7,
                        VisibleAbsorptance = 0.7,
                    };

                case Materials.MetalReflective:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Metal Roof Surface - Highly Reflective",
                        Thickness = 0.0007999999999999979,
                        Conductivity = 45.24971874361766,
                        Density = 7824.017889489713,
                        SpecificHeat = 499.67760800730963,
                        Roughness = Roughness.Smooth,
                        ThermalAbsorptance = 0.75,
                        SolarAbsorptance = 0.45,
                        VisibleAbsorptance = 0.7,
                    };

                case Materials.MoistSoil:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Moist Soil",
                        Thickness = 0.2,
                        Conductivity = 1.0,
                        Density = 1250.0,
                        SpecificHeat = 1252.0,
                        Roughness = Roughness.Rough,
                        ThermalAbsorptance = 0.92,
                        SolarAbsorptance = 0.75,
                        VisibleAbsorptance = 0.75,
                    };

                case Materials.Mud:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Mud",
                        Thickness = 0.2,
                        Conductivity = 1.4,
                        Density = 1840.0,
                        SpecificHeat = 1480.0,
                        Roughness = Roughness.MediumRough,
                        ThermalAbsorptance = 0.95,
                        SolarAbsorptance = 0.8,
                        VisibleAbsorptance = 0.8,
                    };

                case Materials.SolidRock:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Solid Rock",
                        Thickness = 0.2,
                        Conductivity = 3.0,
                        Density = 2700.0,
                        SpecificHeat = 790.0,
                        Roughness = Roughness.MediumRough,
                        ThermalAbsorptance = 0.96,
                        SolarAbsorptance = 0.55,
                        VisibleAbsorptance = 0.55,
                    };

                case Materials.WoodSiding:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Wood Siding",
                        Thickness = 0.010000000000000004,
                        Conductivity = 0.10992643687716327,
                        Density = 544.6212452676245,
                        SpecificHeat = 1209.2198113776988,
                        Roughness = Roughness.MediumSmooth,
                        ThermalAbsorptance = 0.9,
                        SolarAbsorptance = 0.78,
                        VisibleAbsorptance = 0.78,
                    };

                case Materials.Fabric:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Fabric",
                        Thickness = 0.002,
                        Conductivity = 0.06,
                        Density = 500.0,
                        SpecificHeat = 1800.0,
                        Roughness = Roughness.Smooth,
                        ThermalAbsorptance = 0.89,
                        SolarAbsorptance = 0.5,
                        VisibleAbsorptance = 0.5,
                    };

                case Materials.Shrubs:
                    return new OpaqueVegetationMaterial()
                    {
                        Identifier = "Shrubs",
                        Thickness = 0.1,
                        Conductivity = 0.35,
                        Density = 1260,
                        SpecificHeat = 1100,
                        Roughness = Roughness.Rough,
                        SoilThermalAbsorptance = 0.9,
                        SoilSolarAbsorptance = 0.7,
                        SoilVisibleAbsorptance = 0.7,
                        PlantHeight = 0.2,
                        LeafAreaIndex = 2.08,
                        LeafReflectivity = 0.21,
                        LeafEmissivity = 0.95,
                        MinStomatalResist = 180,
                    };

                case Materials.Travertine:
                    return new OpaqueMaterial()
                    {
                        Identifier = "Travertine",
                        Thickness = 0.2,
                        Conductivity = 3.2,
                        Density = 2700.0,
                        SpecificHeat = 790.0,
                        Roughness = Roughness.MediumRough,
                        ThermalAbsorptance = 0.96,
                        SolarAbsorptance = 0.55,
                        VisibleAbsorptance = 0.55,
                    };
                default:
                    return null;
            }
        }
    }
}

