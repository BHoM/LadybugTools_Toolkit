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
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Typology object.")]
        [Input("identifier", "The identifier of the typology.")]
        [Input("shelters", "The shelters of the typology.")]
        [Input("evaporativeCoolingEffect", "A list of hourly-annual dimensionless values by which to adjust the additional of moisture into the air and modify the dry-bulb temperature and relative humidity values. A value 0 means no additional moisure added to air, wheras a value of 1 results in fully moisture saturated air at 100% relative humidity.")]
        [Input("targetWindSpeed", "The hourly target wind speed of the typology, in m/s. This can also contain \"null\" values in which case the EPW file used alongside this object and the porosity of the shelters will be used to determine wind speed - otherwise, any value input here will overwrite those calculated wind speeds.")]
        [Input("radiantTemperatureAdjustment", "A list of values in °C, one-per-hour to adjust the mean radiant temperature by.")]
        [Output("typology", "A Typology object.")]
        public static Typology Typology(
            string identifier = null,
            List<Shelter> shelters = null,
            List<double> evaporativeCoolingEffect = null,
            List<double?> targetWindSpeed = null,
            List<double> radiantTemperatureAdjustment = null
        )
        {
            shelters = shelters ?? new List<Shelter>();

            if ((evaporativeCoolingEffect.Count() == 0 && evaporativeCoolingEffect.Sum() == 0) || evaporativeCoolingEffect == null)
                evaporativeCoolingEffect = Enumerable.Repeat(0.0, 8760).ToList();

            if (evaporativeCoolingEffect.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(evaporativeCoolingEffect)} must be a list of 8760 values.");
                return null;
            }

            if (evaporativeCoolingEffect.Where(x => x < 0 || x > 1).Any())
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(evaporativeCoolingEffect)} must be between 0 and 1.");
                return null;
            }

            if ((targetWindSpeed.Count() == 0 && targetWindSpeed.Sum() == 0) || targetWindSpeed == null)
                targetWindSpeed = Enumerable.Repeat<double?>(null, 8760).ToList();

            if (targetWindSpeed.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(targetWindSpeed)} must be a list of 8760 values.");
                return null;
            }

            if (targetWindSpeed.Where(x => x != null && x.Value < 0).Any())
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(targetWindSpeed)} values must be greater than or equal to 0, or null if not relevant for that hour of the year.");
                return null;
            }

            if ((radiantTemperatureAdjustment.Count() == 0 && radiantTemperatureAdjustment.Sum() == 0) || radiantTemperatureAdjustment == null)
                radiantTemperatureAdjustment = Enumerable.Repeat(0.0, 8760).ToList();

            if (radiantTemperatureAdjustment.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(radiantTemperatureAdjustment)} must be a list of 8760 values.");
                return null;
            }

            if (identifier == null)
            {
                dynamic targetWindSpeedAvg;
                if (targetWindSpeed.Where(x => x.HasValue).Count() == 0)
                {
                    targetWindSpeedAvg = "EPW";
                }
                else
                {
                    targetWindSpeedAvg = targetWindSpeed.Where(x => x.HasValue).Average(x => x.Value);
                }
                if (shelters.Count() == 0)
                {
                    identifier = $"ec{evaporativeCoolingEffect.Average()}_ws{targetWindSpeedAvg}_mrt{radiantTemperatureAdjustment.Average()}";
                }
                else
                {
                    identifier = $"shelters{shelters.Count()}_ec{evaporativeCoolingEffect.Average()}_ws{targetWindSpeedAvg}_mrt{radiantTemperatureAdjustment.Average()}";
                }
                Base.Compute.RecordNote($"This typology has been automatically named \"{identifier}\".");
            }

            return new Typology()
            {
                Name = identifier,
                Shelters = shelters,
                EvaporativeCoolingEffect = evaporativeCoolingEffect,
                TargetWindSpeed = targetWindSpeed,
                RadiantTemperatureAdjustment = radiantTemperatureAdjustment
            };
        }
    }
}
