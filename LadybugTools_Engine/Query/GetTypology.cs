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
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq.Expressions;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get a material object from it's Enum.")]
        [Output("typology", "A typology object to pass into the External Comfort workflow.")]
        public static Typology GetTypology(this Typologies typology)
        {
            if (typology == Typologies.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("A pre-defined typology must be passed in order to return an object.");
            }

            switch (typology)
            {
                case Typologies.Openfield:
                    return new Typology()
                    {
                        Name = "Openfield",
                        Shelters = new List<Shelter>(),
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.Enclosed:
                    return new Typology()
                    {
                        Name = "Enclosed",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        360
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.PorousEnclosure:
                    return new Typology()
                    {
                        Name = "Porous enclosure",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0.5,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        360
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SkyShelter:
                    return new Typology()
                    {
                        Name = "Sky-shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        360
                                    },
                                    AltitudeRange = new List < double > () {
                                        45,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.FrittedSkyShelter:
                    return new Typology()
                    {
                        Name = "Fritted sky-shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0.5,
                                    RadiationPorosity = 0.5,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        360
                                    },
                                    AltitudeRange = new List < double > () {
                                        45,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NearWater:
                    return new Typology()
                    {
                        Name = "Near water",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 1,
                                    RadiationPorosity = 1,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        0
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        0
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.15,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.Misting:
                    return new Typology()
                    {
                        Name = "Misting",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 1,
                                    RadiationPorosity = 1,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        0
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        0
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.3,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.Pdec:
                    return new Typology()
                    {
                        Name = "PDEC",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 1,
                                    RadiationPorosity = 1,
                                    AzimuthRange = new List < double > () {
                                        0,
                                        0
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        0
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.7,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthShelter:
                    return new Typology()
                    {
                        Name = "North shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        337.5,
                                        22.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NortheastShelter:
                    return new Typology()
                    {
                        Name = "Northeast shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        22.5,
                                        67.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastShelter:
                    return new Typology()
                    {
                        Name = "East shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        67.5,
                                        112.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SoutheastShelter:
                    return new Typology()
                    {
                        Name = "Southeast shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        112.5,
                                        157.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthShelter:
                    return new Typology()
                    {
                        Name = "South shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        157.5,
                                        202.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthwestShelter:
                    return new Typology()
                    {
                        Name = "Southwest shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        202.5,
                                        247.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.WestShelter:
                    return new Typology()
                    {
                        Name = "West shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        247.5,
                                        292.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthwestShelter:
                    return new Typology()
                    {
                        Name = "Northwest shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        292.5,
                                        337.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        70
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "North shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        337.5,
                                        22.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NortheastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Northeast shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        22.5,
                                        67.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "East shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        67.5,
                                        112.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SoutheastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Southeast shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        112.5,
                                        157.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "South shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        157.5,
                                        202.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthwestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Southwest shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        202.5,
                                        247.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.WestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "West shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        247.5,
                                        292.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthwestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Northwest shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                    RadiationPorosity = 0,
                                    AzimuthRange = new List < double > () {
                                        292.5,
                                        337.5
                                    },
                                    AltitudeRange = new List < double > () {
                                        0,
                                        90
                                    },
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastWestShelter:
                    return new Typology()
                    {
                        Name = "East-west shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                    WindPorosity = 0,
                                        RadiationPorosity = 0,
                                        AzimuthRange = new List < double > () {
                                            67.5,
                                            112.5
                                        },
                                        AltitudeRange = new List < double > () {
                                            0,
                                            70
                                        },
                                },
                                new Shelter() {
                                    WindPorosity = 0,
                                        RadiationPorosity = 0,
                                        AzimuthRange = new List < double > () {
                                            247.5,
                                            292.5
                                        },
                                        AltitudeRange = new List < double > () {
                                            0,
                                            70
                                        },
                                },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastWestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "East-west shelter (with canopy)",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                    WindPorosity = 0,
                                        RadiationPorosity = 0,
                                        AzimuthRange = new List < double > () {
                                            67.5,
                                            112.5
                                        },
                                        AltitudeRange = new List < double > () {
                                            0,
                                            90
                                        },
                                },
                                new Shelter() {
                                    WindPorosity = 0,
                                        RadiationPorosity = 0,
                                        AzimuthRange = new List < double > () {
                                            247.5,
                                            292.5
                                        },
                                        AltitudeRange = new List < double > () {
                                            0,
                                            90
                                        },
                                },
                        },
                        EvaporativeCoolingEffectiveness = 0.0,
                        WindSpeedAdjustment = 1,
                    };
                default:
                    return null;
            }
        }
    }
}