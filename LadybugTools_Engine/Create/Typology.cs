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
        [Description("Create a Typology object for external comfort simulations.")]
        [Input("shelters", "A list of shelters modifying exposure to the elements.")]
        [Input("name", "Optional name of this typology.")]
        [Input("evaporativeCoolingEffect", "An amount of evaporative cooling to add to results calculated by this typology. Defaults to 0, corresponding to no additional water added to the air. A value of 0.1 might represent a nearby body of water, and a value of 0.3 might represent a nearby misting device.")]
        [Input("windSpeedMultiplier", "A factor to multiply wind speed by. Defaults to 1. Can be used to account for wind speed reduction due to sheltering not accounted for by shelter objects, or to approximate effects of acceleration. Default = 1. Set to 0 for still air conditions.")]
        [Input("radiantTemperatureAdjustment", "A change in MRT to be applied. Defaults to 0. A positive value will increase the MRT and a negative value will decrease it.")]
        [Output("typology", "Typology object.")]
        public static Typology Typology(List<Shelter> shelters = null, string name = "", double evaporativeCoolingEffect = 0, double windSpeedMultiplier = 1, double radiantTemperatureAdjustment = 0)
        {
            if (!evaporativeCoolingEffect.IsBetween(0, 1))
            {
                Base.Compute.RecordError($"{nameof(evaporativeCoolingEffect)} must be between 0 and 1, but is currently {evaporativeCoolingEffect}.");
                return null;
            }

            if (windSpeedMultiplier < 0)
            {
                Base.Compute.RecordError($"{nameof(windSpeedMultiplier)} must be greater than or equal to 0, but is currently {windSpeedMultiplier}.");
                return null;
            }

            if (!radiantTemperatureAdjustment.IsBetween(-10, 10))
            {
                Base.Compute.RecordError($"{nameof(radiantTemperatureAdjustment)} must be between -10 and 10, but is currently {radiantTemperatureAdjustment}.");
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
                rtnName = $"ec{evaporativeCoolingEffect}_ws{windSpeedMultiplier}_mrt{radiantTemperatureAdjustment}";
                Base.Compute.RecordNote($"This typology has been automatically named {rtnName}. This can be overriden with the 'name' parameter.");
            }
            else if (rtnName == "")
            {
                rtnName = $"shelters{shelterCount}_ec{evaporativeCoolingEffect}_ws{windSpeedMultiplier}_mrt{radiantTemperatureAdjustment}";
                Base.Compute.RecordNote($"This typology has been automatically named {rtnName}. This can be overriden with the 'name' parameter.");
            }

            return new Typology()
            {
                Name = name,
                Shelters = shelters.Where(s => s != null).ToList(),
                EvaporativeCoolingEffect = Enumerable.Repeat(evaporativeCoolingEffect, 8760).ToList(),
                WindSpeedMultiplier = Enumerable.Repeat(windSpeedMultiplier, 8760).ToList(),
                RadiantTemperatureAdjustment = Enumerable.Repeat(radiantTemperatureAdjustment, 8760).ToList(),
            };
        }

        private static bool IsBetween(this double val, double min, double max)
        {
            return (val >= min && val <= max);
        }
    }
}

