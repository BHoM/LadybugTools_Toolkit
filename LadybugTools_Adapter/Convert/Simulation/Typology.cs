/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Linq;
using System.Text;
using BH.oM.Base;
using BH.oM.LadybugTools;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static oM.LadybugTools.Typology ToTypology(Dictionary<string, object> oldObject)
        {
            string identifier = "";
            List<Shelter> shelters = new List<Shelter>();
            List<double> evaporativeCoolingEffect = Enumerable.Repeat(0.0, 8760).ToList();
            List<double?> targetWindSpeed = Enumerable.Repeat<double?>(null, 8760).ToList();
            double windSpeedMultiplier;
            List<double> radiantTemperatureAdjustment = Enumerable.Repeat(0.0, 8760).ToList();

            try
            {
                identifier = (string)oldObject["identifier"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the identifier of the typology. Returning default value (\"\").\n The error: {ex}");
            }

            try
            {
                foreach (var shelter in oldObject["shelters"] as List<object>)
                    if (shelter.GetType() == typeof(CustomObject))
                        shelters.Add(ToShelter(((CustomObject)shelter).CustomData));
                    else
                        shelters.Add(ToShelter((Dictionary<string, object>)shelter));
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing one of the shelters of the typology. Returning any shelters that could be parsed.\n The error: {ex}");
            }

            try
            {
                List<double> values = new List<double>();
                foreach (object value in oldObject["evaporative_cooling_effect"] as List<object>)
                    values.Add(double.Parse(value.ToString()));

                evaporativeCoolingEffect = values;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the evaporative cooling effect of the typology. Returning a list of 0.0s of length 8760.\n The error: {ex}");
            }

            try
            {
                List<double?> values = new List<double?>();
                foreach (object value in oldObject["target_wind_speed"] as List<object>)
                {
                    if (value == null)
                        values.Add(null);
                    else if (double.TryParse(value.ToString(), out double result))
                        values.Add(result);
                    else
                        values.Add(null);
                }

                targetWindSpeed = values;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the target wind speed of the typology. Returning a list of nulls of length 8760.\n The error: {ex}");
            }

            if (!double.TryParse(oldObject["wind_speed_multiplier"].ToString(), out windSpeedMultiplier))
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the wind speed multiplier of the typology. Returning default of 1 instead.");
            }

            try
            {
                List<double> values = new List<double>();
                foreach (object value in oldObject["radiant_temperature_adjustment"] as List<object>)
                    values.Add(double.Parse(value.ToString()));

                radiantTemperatureAdjustment = values;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the radiant temperature adjustment of the typology. Returning a list of 0.0s of length 8760.\n The error: {ex}");
            }

            return new Typology()
{
                Name = identifier,
                Shelters = shelters,
                EvaporativeCoolingEffect = evaporativeCoolingEffect,
                TargetWindSpeed = targetWindSpeed,
                WindSpeedMultiplier = windSpeedMultiplier,
                RadiantTemperatureAdjustment = radiantTemperatureAdjustment
            };
        }

        public static string FromTypology(oM.LadybugTools.Typology typology)
    {
            string identifier = typology.Name;
            string shelters = "[" + string.Join(", ", typology.Shelters.Select(s => FromShelter(s)).ToList()) + "]";
            string evaporativeCoolingEffect = "[" + string.Join(", ", typology.EvaporativeCoolingEffect) + "]";
            string targetWindSpeed = "[" + string.Join(", ", typology.TargetWindSpeed.Select(x => x.ToString()).Select(x => x == "" ? x = "null": x)) + "]";
            string radiantTemperatureAdjustment = "[" + string.Join(", ", typology.RadiantTemperatureAdjustment) + "]";
            return @"{""type"": ""Typology"", " + 
                $@"""identifier"": ""{identifier}"", " + 
                $@"""shelters"": {shelters}, " + 
                $@"""evaporative_cooling_effect"": {evaporativeCoolingEffect}, " + 
                $@"""target_wind_speed"": {targetWindSpeed}, " + 
                $@"""wind_speed_multiplier"": {typology.WindSpeedMultiplier}, " +
                $@"""radiant_temperature_adjustment"": {radiantTemperatureAdjustment}" + "}";
        }
    }
}


