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

using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ExternalComfort ToExternalComfort(Dictionary<string, object> oldObject)
        {
            SimulationResult simulationResult = new SimulationResult();
            Typology typology = new Typology();
            List<HourlyContinuousCollection> simulatedProperties = new List<HourlyContinuousCollection>();

            List<string> properties = new List<string>()
            {
                "dry_bulb_temperature",
                "relative_humidity",
                "wind_speed",
                "mean_radiant_temperature",
                "universal_thermal_climate_index"
            };

            try
            {
                if (oldObject["simulation_result"].GetType() == typeof(CustomObject))
                    oldObject["simulation_result"] = ((CustomObject)oldObject["simulation_result"]).CustomData;
                simulationResult = ToSimulationResult(oldObject["simulation_result"] as Dictionary<string, object>);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the SimulationResult of the ExternalComfort. Returning a default SimulationResult. \n The error: {ex}");
            }

            try
            {
                if (oldObject["typology"].GetType() == typeof(CustomObject))
                    oldObject["typology"] = ((CustomObject)oldObject["typology"]).CustomData;
                typology = ToTypology(oldObject["typology"] as Dictionary<string, object>);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the Typology of the ExternalComfort. Returning a default Typology. \n The error: {ex}");
            }

            foreach (string property in properties)
            {
                if (oldObject.ContainsKey(property))
                {
                    try
                    {
                        if (oldObject[property].GetType() == typeof(CustomObject))
                            oldObject[property] = ((CustomObject)oldObject[property]).CustomData;
                        simulatedProperties.Add(ToHourlyContinuousCollection(oldObject[property] as Dictionary<string, object>));
                    }
                    catch (Exception ex)
                    {
                        BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the collection {property} of the ExternalComfort. Returning an empty collection in its place.\n The error: {ex}");
                        simulatedProperties.Add(new HourlyContinuousCollection() { Values = Enumerable.Repeat<string>(null, 8760).ToList() });
                    }
                }
                else
                {
                    BH.Engine.Base.Compute.RecordError($"The incoming json for ExternalComfort does not contain the key: {property}. Returning an empty collection in its place.");
                    simulatedProperties.Add(new HourlyContinuousCollection() { Values = Enumerable.Repeat<string>(null, 8760).ToList() });
                }
            }

            return new ExternalComfort(simulationResult, typology, simulatedProperties[0], simulatedProperties[1], simulatedProperties[2], simulatedProperties[3], simulatedProperties[4]);
        }

        public static string FromExternalComfort(ExternalComfort externalComfort)
        {
            string type = "\"type\": \"ExternalComfort\", ";
            string simulationResult = $"\"simulation_result\": {FromSimulationResult(externalComfort.SimulationResult)}, ";
            string typology = $"\"typology\": {FromTypology(externalComfort.Typology)}";
            List<string> properties = new List<string>();

            if (externalComfort.DryBulbTemperature != null)
                properties.Add("\"dry_bulb_temperature\": " + FromHourlyContinuousCollection(externalComfort.DryBulbTemperature));

            if (externalComfort.RelativeHumidity != null)
                properties.Add("\"relative_humidity\": " + FromHourlyContinuousCollection(externalComfort.RelativeHumidity));

            if (externalComfort.WindSpeed != null)
                properties.Add("\"wind_speed\": " + FromHourlyContinuousCollection(externalComfort.WindSpeed));

            if (externalComfort.MeanRadiantTemperature != null)
                properties.Add("\"mean_radiant_temperature\": " + FromHourlyContinuousCollection(externalComfort.MeanRadiantTemperature));

            if (externalComfort.UniversalThermalClimateIndex != null)
                properties.Add("\"universal_thermal_climate_index\": " + FromHourlyContinuousCollection(externalComfort.UniversalThermalClimateIndex));

            if (properties.Count > 0)
                properties[0] = ", " + properties[0];
            string simulatedProperties = string.Join(", ", properties);
            return "{" + type + simulationResult + typology + simulatedProperties + "}";
        }
    }
}


