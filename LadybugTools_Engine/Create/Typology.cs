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
using BH.oM.Environment.Elements;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Threading;
using System.Linq;
using BH.Engine.Base;
using BH.Engine.Environment;
using System.Runtime.CompilerServices;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Typology object for external comfort simulations.")]
        [Input("shelters", "Shelter objects that provide shading from the sun and wind.")]
        [Input("name", "Optional name of this typology.")]
        [Input("evaporativeCoolingEffectiveness", "The proportion of evaporative cooling to add to this ExternalComfortTypology. Must be between 0 and 1. Default = 0, corresponding to no additional water added to the air. A value of 0.1 might represent a nearby body of water, and a value of 0.3 might represent a nearby misting device.")]
        [Input("windSpeedAdjustment", "Multiply weatherfile wind speed by this value. Default = 1. Set to 0 to set wind to still.")]
        [Output("typology", "Typology object.")]
        public static Typology Typology(List<Shelter> shelters = null, string name = "", double evaporativeCoolingEffectiveness = 0, double windSpeedAdjustment = 1)
        {
            if (!evaporativeCoolingEffectiveness.IsBetween(0, 1))
            {
                Base.Compute.RecordError($"{nameof(evaporativeCoolingEffectiveness)} must be between 0 and 1, but is actually {evaporativeCoolingEffectiveness}.");
                return null;
            }

            if (windSpeedAdjustment < 0)
            {
                Base.Compute.RecordError($"{nameof(windSpeedAdjustment)} must be greater than or equal to 0, but is actually {windSpeedAdjustment}.");
                return null;
            }

            if (shelters == null)
            {
                shelters = new List<Shelter>();
            }

            string rtnName = name;
            int shelterCount = shelters.Count(s => s != null);
            if (rtnName == "" && shelterCount == 0)
            {
                rtnName = $"Openfield_ec{evaporativeCoolingEffectiveness}_ws{windSpeedAdjustment}";
                Base.Compute.RecordNote($"This typology has been automatically named {rtnName}. This can be overriden with the 'name' parameter.");
            }
            else if (rtnName == "")
            {
                rtnName = $"Shelters{shelterCount}_ec{evaporativeCoolingEffectiveness}_ws{windSpeedAdjustment}";
                Base.Compute.RecordNote($"This typology has been automatically named {rtnName}. This can be overriden with the 'name' parameter.");
            }

            return new Typology()
            {
                Name = name,
                Shelters = shelters.Where(s => s != null).ToList(),
                EvaporativeCoolingEffectiveness = evaporativeCoolingEffectiveness,
                WindSpeedAdjustment = windSpeedAdjustment
            };
        }

        private static bool IsBetween(this double val, double min, double max)
        {
            return (val >= min && val <= max);
        }
    }
}

