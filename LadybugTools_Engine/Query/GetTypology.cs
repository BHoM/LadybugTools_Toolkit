/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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
        [Input("typology", "An enum value representing a pre-defined Typology object.")]
        [Output("typology", "A Typology object to pass into the External Comfort workflow.")]
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
                        Shelters = new List<Shelter>()
                        {
                        },
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 0},
                                    new List<double>() {-1, 1, 0},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {1, 1, 5},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.0, -0.9999999999999999, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 5.0},
                                    new List<double>() {1.0, -0.9999999999999999, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 5.0},
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 5.0},
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {250, 0, 3.5},
                                    new List<double>() {234.923155, 85.505036, 3.5},
                                    new List<double>() {191.511111, 160.696902, 3.5},
                                    new List<double>() {125, 216.506351, 3.5},
                                    new List<double>() {43.412044, 246.201938, 3.5},
                                    new List<double>() {-43.412044, 246.201938, 3.5},
                                    new List<double>() {-125.0, 216.506351, 3.5},
                                    new List<double>() {-191.511111, 160.696902, 3.5},
                                    new List<double>() {-234.923155, 85.505036, 3.5},
                                    new List<double>() {-250, 0, 3.5},
                                    new List<double>() {-234.923155, -85.505036, 3.5},
                                    new List<double>() {-191.511111, -160.696902, 3.5},
                                    new List<double>() {-125, -216.506351, 3.5},
                                    new List<double>() {-43.412044, -246.201938, 3.5},
                                    new List<double>() {43.412044, -246.201938, 3.5},
                                    new List<double>() {125.0, -216.506351, 3.5},
                                    new List<double>() {191.511111, -160.696902, 3.5},
                                    new List<double>() {234.923155, -85.505036, 3.5},
                                }
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
                                WindPorosity = 0.5,
                                RadiationPorosity = 0.5,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 0},
                                    new List<double>() {-1, 1, 0},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {1, 1, 5},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0.5,
                                RadiationPorosity = 0.5,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.0, -0.9999999999999999, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 5.0},
                                    new List<double>() {1.0, -0.9999999999999999, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0.5,
                                RadiationPorosity = 0.5,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 5.0},
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0.5,
                                RadiationPorosity = 0.5,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 5.0},
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0.5,
                                RadiationPorosity = 0.5,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {250, 0, 3.5},
                                    new List<double>() {234.923155, 85.505036, 3.5},
                                    new List<double>() {191.511111, 160.696902, 3.5},
                                    new List<double>() {125, 216.506351, 3.5},
                                    new List<double>() {43.412044, 246.201938, 3.5},
                                    new List<double>() {-43.412044, 246.201938, 3.5},
                                    new List<double>() {-125.0, 216.506351, 3.5},
                                    new List<double>() {-191.511111, 160.696902, 3.5},
                                    new List<double>() {-234.923155, 85.505036, 3.5},
                                    new List<double>() {-250, 0, 3.5},
                                    new List<double>() {-234.923155, -85.505036, 3.5},
                                    new List<double>() {-191.511111, -160.696902, 3.5},
                                    new List<double>() {-125, -216.506351, 3.5},
                                    new List<double>() {-43.412044, -246.201938, 3.5},
                                    new List<double>() {43.412044, -246.201938, 3.5},
                                    new List<double>() {125.0, -216.506351, 3.5},
                                    new List<double>() {191.511111, -160.696902, 3.5},
                                    new List<double>() {234.923155, -85.505036, 3.5},
                                }
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {250, 0, 3.5},
                                    new List<double>() {234.923155, 85.505036, 3.5},
                                    new List<double>() {191.511111, 160.696902, 3.5},
                                    new List<double>() {125, 216.506351, 3.5},
                                    new List<double>() {43.412044, 246.201938, 3.5},
                                    new List<double>() {-43.412044, 246.201938, 3.5},
                                    new List<double>() {-125.0, 216.506351, 3.5},
                                    new List<double>() {-191.511111, 160.696902, 3.5},
                                    new List<double>() {-234.923155, 85.505036, 3.5},
                                    new List<double>() {-250, 0, 3.5},
                                    new List<double>() {-234.923155, -85.505036, 3.5},
                                    new List<double>() {-191.511111, -160.696902, 3.5},
                                    new List<double>() {-125, -216.506351, 3.5},
                                    new List<double>() {-43.412044, -246.201938, 3.5},
                                    new List<double>() {43.412044, -246.201938, 3.5},
                                    new List<double>() {125.0, -216.506351, 3.5},
                                    new List<double>() {191.511111, -160.696902, 3.5},
                                    new List<double>() {234.923155, -85.505036, 3.5},
                                }
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {250, 0, 3.5},
                                    new List<double>() {234.923155, 85.505036, 3.5},
                                    new List<double>() {191.511111, 160.696902, 3.5},
                                    new List<double>() {125, 216.506351, 3.5},
                                    new List<double>() {43.412044, 246.201938, 3.5},
                                    new List<double>() {-43.412044, 246.201938, 3.5},
                                    new List<double>() {-125.0, 216.506351, 3.5},
                                    new List<double>() {-191.511111, 160.696902, 3.5},
                                    new List<double>() {-234.923155, 85.505036, 3.5},
                                    new List<double>() {-250, 0, 3.5},
                                    new List<double>() {-234.923155, -85.505036, 3.5},
                                    new List<double>() {-191.511111, -160.696902, 3.5},
                                    new List<double>() {-125, -216.506351, 3.5},
                                    new List<double>() {-43.412044, -246.201938, 3.5},
                                    new List<double>() {43.412044, -246.201938, 3.5},
                                    new List<double>() {125.0, -216.506351, 3.5},
                                    new List<double>() {191.511111, -160.696902, 3.5},
                                    new List<double>() {234.923155, -85.505036, 3.5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NearWater:
                    return new Typology()
                    {
                        Name = "Near water",
                        Shelters = new List<Shelter>()
                        {
                        },
                        EvaporativeCoolingEffectiveness = 0.15,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.Misting:
                    return new Typology()
                    {
                        Name = "Misting",
                        Shelters = new List<Shelter>()
                        {
                        },
                        EvaporativeCoolingEffectiveness = 0.3,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.Pdec:
                    return new Typology()
                    {
                        Name = "PDEC",
                        Shelters = new List<Shelter>()
                        {
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 0},
                                    new List<double>() {-1, 1, 0},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {1, 1, 5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 0.0},
                                    new List<double>() {0.0, 1.4142135623730951, 0.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.0, -0.9999999999999999, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 5.0},
                                    new List<double>() {1.0, -0.9999999999999999, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.1102230246251565e-16, -1.414213562373095, 0.0},
                                    new List<double>() {1.414213562373095, 1.1102230246251565e-16, 0.0},
                                    new List<double>() {1.414213562373095, 1.1102230246251565e-16, 5.0},
                                    new List<double>() {1.1102230246251565e-16, -1.414213562373095, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 5.0},
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.4142135623730951, -2.220446049250313e-16, 0.0},
                                    new List<double>() {2.220446049250313e-16, -1.4142135623730951, 0.0},
                                    new List<double>() {2.220446049250313e-16, -1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, -2.220446049250313e-16, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 5.0},
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
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
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-3.3306690738754696e-16, 1.414213562373095, 0.0},
                                    new List<double>() {-1.414213562373095, -3.3306690738754696e-16, 0.0},
                                    new List<double>() {-1.414213562373095, -3.3306690738754696e-16, 5.0},
                                    new List<double>() {-3.3306690738754696e-16, 1.414213562373095, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "North shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 0},
                                    new List<double>() {-1, 1, 0},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {1, 1, 5},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 5},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {-1, -1, 5},
                                    new List<double>() {1, -1, 5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NortheastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Northeast shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 0.0},
                                    new List<double>() {0.0, 1.4142135623730951, 0.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, -1.4142135623730951, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "East shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.0, -0.9999999999999999, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 0.0},
                                    new List<double>() {0.9999999999999999, 1.0, 5.0},
                                    new List<double>() {1.0, -0.9999999999999999, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 5},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {-1, -1, 5},
                                    new List<double>() {1, -1, 5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SoutheastShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Southeast shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.1102230246251565e-16, -1.414213562373095, 0.0},
                                    new List<double>() {1.414213562373095, 1.1102230246251565e-16, 0.0},
                                    new List<double>() {1.414213562373095, 1.1102230246251565e-16, 5.0},
                                    new List<double>() {1.1102230246251565e-16, -1.414213562373095, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, -1.4142135623730951, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "South shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 0.0},
                                    new List<double>() {1.0000000000000002, -0.9999999999999999, 5.0},
                                    new List<double>() {-0.9999999999999999, -1.0000000000000002, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 5},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {-1, -1, 5},
                                    new List<double>() {1, -1, 5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.SouthwestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Southwest shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.4142135623730951, -2.220446049250313e-16, 0.0},
                                    new List<double>() {2.220446049250313e-16, -1.4142135623730951, 0.0},
                                    new List<double>() {2.220446049250313e-16, -1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, -2.220446049250313e-16, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, -1.4142135623730951, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.WestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "West shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 0.0},
                                    new List<double>() {-0.9999999999999998, -1.0000000000000002, 5.0},
                                    new List<double>() {-1.0000000000000002, 0.9999999999999998, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1, 1, 5},
                                    new List<double>() {-1, 1, 5},
                                    new List<double>() {-1, -1, 5},
                                    new List<double>() {1, -1, 5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthwestShelterWithCanopy:
                    return new Typology()
                    {
                        Name = "Northwest shelter with canopy",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-3.3306690738754696e-16, 1.414213562373095, 0.0},
                                    new List<double>() {-1.414213562373095, -3.3306690738754696e-16, 0.0},
                                    new List<double>() {-1.414213562373095, -3.3306690738754696e-16, 5.0},
                                    new List<double>() {-3.3306690738754696e-16, 1.414213562373095, 5.0},
                                }
                            },
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, 1.4142135623730951, 5.0},
                                    new List<double>() {-1.4142135623730951, 0.0, 5.0},
                                    new List<double>() {0.0, -1.4142135623730951, 5.0},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthsouthLinearShelter:
                    return new Typology()
                    {
                        Name = "North-south linear overhead shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {2, -1000, 3.5},
                                    new List<double>() {2, 1000, 3.5},
                                    new List<double>() {-2, 1000, 3.5},
                                    new List<double>() {-2, -1000, 3.5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NortheastSouthwestLinearShelter:
                    return new Typology()
                    {
                        Name = "Northeast-southwest linear overhead shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-705.6925676241744, -708.5209947489207, 3.5},
                                    new List<double>() {708.5209947489207, 705.6925676241744, 3.5},
                                    new List<double>() {705.6925676241744, 708.5209947489207, 3.5},
                                    new List<double>() {-708.5209947489207, -705.6925676241744, 3.5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.EastWestLinearShelter:
                    return new Typology()
                    {
                        Name = "East-west linear overhead shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-1000.0, -2.0000000000000613, 3.5},
                                    new List<double>() {1000.0, -1.9999999999999387, 3.5},
                                    new List<double>() {1000.0, 2.0000000000000613, 3.5},
                                    new List<double>() {-1000.0, 1.9999999999999387, 3.5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                case Typologies.NorthwestSoutheastLinearShelter:
                    return new Typology()
                    {
                        Name = "Northwest-southeast linear overhead shelter",
                        Shelters = new List<Shelter>() {
                            new Shelter() {
                                WindPorosity = 0,
                                RadiationPorosity = 0,
                                Vertices = new List<List<double>>() {
                                    new List<double>() {-708.5209947489207, 705.6925676241743, 3.5},
                                    new List<double>() {705.6925676241744, -708.5209947489205, 3.5},
                                    new List<double>() {708.5209947489207, -705.6925676241743, 3.5},
                                    new List<double>() {-705.6925676241744, 708.5209947489205, 3.5},
                                }
                            },
                        },
                        EvaporativeCoolingEffectiveness = 0,
                        WindSpeedAdjustment = 1,
                    };
                default:
                    return null;
            }
        }
    }
}
