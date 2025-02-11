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

using BH.Engine.Adapter;
using BH.Engine.Base;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static SimulationResult ToSimulationResult(Dictionary<string, object> oldObject)
        {
            string epwFile = "";
            IEnergyMaterialOpaque groundMaterial = new EnergyMaterial();
            IEnergyMaterialOpaque shadeMaterial = new EnergyMaterial();
            string name = "";
            List<HourlyContinuousCollection> simulatedProperties = new List<HourlyContinuousCollection>();
            List<string> properties = new List<string>()
            {
                "shaded_down_temperature",
                "shaded_up_temperature",
                "shaded_radiant_temperature",
                "shaded_longwave_mean_radiant_temperature_delta",
                "shaded_shortwave_mean_radiant_temperature_delta",
                "shaded_mean_radiant_temperature",
                "unshaded_down_temperature",
                "unshaded_up_temperature",
                "unshaded_radiant_temperature",
                "unshaded_longwave_mean_radiant_temperature_delta",
                "unshaded_shortwave_mean_radiant_temperature_delta",
                "unshaded_mean_radiant_temperature"
            };

            try
            {
                epwFile = (string)oldObject["epw_file"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred during parsing of the epw file path of the SimulationResult. Returning default value (\"\").\n The error: {ex}");
            }

            try
            {
                if (oldObject["ground_material"].GetType() == typeof(CustomObject))
                    oldObject["ground_material"] = ((CustomObject)oldObject["ground_material"]).CustomData;
                switch ((oldObject["ground_material"] as Dictionary<string, object>)["type"])
                {
                    case "EnergyMaterial":
                        groundMaterial = ToEnergyMaterial(oldObject["ground_material"] as Dictionary<string, object>);
                        break;
                    case "EnergyMaterialVegetation":
                        groundMaterial = ToEnergyMaterialVegetation(oldObject["ground_material"] as Dictionary<string, object>);
                        break;
                    default:
                        BH.Engine.Base.Compute.RecordError($"The ground material given is not an IEnergyMaterialOpaque but {(oldObject["ground_material"] as Dictionary<string, object>)["type"]}, so cannot be explicitly converted to an EnergyMaterialOpaque. Trying convert to EnergyMaterial...");
                        groundMaterial = ToEnergyMaterial(oldObject["ground_material"] as Dictionary<string, object>);
                        break;
                }
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when deserialising the ground material of the SimulationResult. returning a default EnergyMaterial.\n The error: {ex}");
            }

            try
            {
                name = (string)oldObject["identifier"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the name of the SimulationResult. Returning default value (\"\").\n The error: {ex}");
            }

            try
            {
                if (oldObject["shade_material"].GetType() == typeof(CustomObject))
                    oldObject["shade_material"] = ((CustomObject)oldObject["shade_material"]).CustomData;
                switch ((oldObject["shade_material"] as Dictionary<string, object>)["type"])
                {
                    case "EnergyMaterial":
                        shadeMaterial = ToEnergyMaterial(oldObject["shade_material"] as Dictionary<string, object>);
                        break;
                    case "EnergyMaterialVegetation":
                        shadeMaterial = ToEnergyMaterialVegetation(oldObject["shade_material"] as Dictionary<string, object>);
                        break;
                    default:
                        BH.Engine.Base.Compute.RecordWarning($"The shade material given is not an IEnergyMaterialOpaque but {(oldObject["shade_material"] as Dictionary<string, object>)["type"]}, so cannot be explicitly converted to an EnergyMaterialOpaque. Trying convert to EnergyMaterial...");
                        shadeMaterial = ToEnergyMaterial(oldObject["shade_material"] as Dictionary<string, object>);
                        break;
                }
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when deserialising the shade material of the SimulationResult. returning a default EnergyMaterial.\n The error: {ex}");
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
                        BH.Engine.Base.Compute.RecordError($"An error occurred while parsing the collection {property} of the SimulationResult. Returning an empty collection in its place.\n The error: {ex}");
                        simulatedProperties.Add(new HourlyContinuousCollection() { Values = Enumerable.Repeat<string>(null, 8760).ToList() });
                    }
                }
                else
                {
                    BH.Engine.Base.Compute.RecordError($"The incoming json does not contain the key: {property}. Returning an empty collection in its place.");
                    simulatedProperties.Add(new HourlyContinuousCollection() { Values = Enumerable.Repeat<string>(null, 8760).ToList() });
                }
            }

            return new SimulationResult()
            {
                EpwFile = new FileSettings() { FileName = System.IO.Path.GetFileName(epwFile), Directory = System.IO.Path.GetDirectoryName(epwFile) },
                GroundMaterial = groundMaterial,
                ShadeMaterial = shadeMaterial,
                Name = name,
                ShadedDownTemperature = simulatedProperties[0],
                ShadedUpTemperature = simulatedProperties[1],
                ShadedRadiantTemperature = simulatedProperties[2],
                ShadedLongwaveMeanRadiantTemperatureDelta = simulatedProperties[3],
                ShadedShortwaveMeanRadiantTemperatureDelta = simulatedProperties[4],
                ShadedMeanRadiantTemperature = simulatedProperties[5],
                UnshadedDownTemperature = simulatedProperties[6],
                UnshadedUpTemperature = simulatedProperties[7],
                UnshadedRadiantTemperature = simulatedProperties[8],
                UnshadedLongwaveMeanRadiantTemperatureDelta = simulatedProperties[9],
                UnshadedShortwaveMeanRadiantTemperatureDelta = simulatedProperties[10],
                UnshadedMeanRadiantTemperature = simulatedProperties[11]
            };
        }

        public static string FromSimulationResult(SimulationResult simulationResult)
        {
            string type = $"\"type\": \"SimulationResult\", ";
            string epwFile = $"\"epw_file\": \"{simulationResult.EpwFile.GetFullFileName().Replace("\\", "\\\\")}\", ";
            string groundMaterial = $"\"ground_material\": {FromBHoM(simulationResult.GroundMaterial)}, ";
            string shadeMaterial = $"\"shade_material\": {FromBHoM(simulationResult.ShadeMaterial)}, ";
            string name = $"\"identifier\": \"{simulationResult.Name}\"";
            List<string> properties = new List<string>();

            if (simulationResult.ShadedDownTemperature != null)
                properties.Add("\"shaded_down_temperature\": " + FromHourlyContinuousCollection(simulationResult.ShadedDownTemperature));

            if (simulationResult.ShadedUpTemperature != null)
                properties.Add("\"shaded_up_temperature\": " + FromHourlyContinuousCollection(simulationResult.ShadedUpTemperature));

            if (simulationResult.ShadedRadiantTemperature != null)
                properties.Add("\"shaded_radiant_temperature\": " + FromHourlyContinuousCollection(simulationResult.ShadedRadiantTemperature));

            if (simulationResult.ShadedLongwaveMeanRadiantTemperatureDelta != null)
                properties.Add("\"shaded_longwave_mean_radiant_temperature_delta\": " + FromHourlyContinuousCollection(simulationResult.ShadedLongwaveMeanRadiantTemperatureDelta));

            if (simulationResult.ShadedShortwaveMeanRadiantTemperatureDelta != null)
                properties.Add("\"shaded_shortwave_mean_radiant_temperature_delta\": " + FromHourlyContinuousCollection(simulationResult.ShadedShortwaveMeanRadiantTemperatureDelta));

            if (simulationResult.ShadedMeanRadiantTemperature != null)
                properties.Add("\"shaded_mean_radiant_temperature\": " + FromHourlyContinuousCollection(simulationResult.ShadedMeanRadiantTemperature));

            if (simulationResult.UnshadedUpTemperature != null)
                properties.Add("\"unshaded_up_temperature\": " + FromHourlyContinuousCollection(simulationResult.UnshadedUpTemperature));

            if (simulationResult.UnshadedDownTemperature != null)
                properties.Add("\"unshaded_down_temperature\": " + FromHourlyContinuousCollection(simulationResult.UnshadedDownTemperature));

            if (simulationResult.UnshadedRadiantTemperature != null)
                properties.Add("\"unshaded_radiant_temperature\": " + FromHourlyContinuousCollection(simulationResult.UnshadedRadiantTemperature));

            if (simulationResult.UnshadedLongwaveMeanRadiantTemperatureDelta != null)
                properties.Add("\"unshaded_longwave_mean_radiant_temperature_delta\": " + FromHourlyContinuousCollection(simulationResult.UnshadedLongwaveMeanRadiantTemperatureDelta));

            if (simulationResult.UnshadedShortwaveMeanRadiantTemperatureDelta != null)
                properties.Add("\"unshaded_shortwave_mean_radiant_temperature_delta\": " + FromHourlyContinuousCollection(simulationResult.UnshadedShortwaveMeanRadiantTemperatureDelta));

            if (simulationResult.UnshadedMeanRadiantTemperature != null)
                properties.Add("\"unshaded_mean_radiant_temperature\": " + FromHourlyContinuousCollection(simulationResult.UnshadedMeanRadiantTemperature));

            if (properties.Count > 0)
                properties[0] = ", " + properties[0];

            string simulatedProperties = string.Join(", ", properties);

            return "{" + type + epwFile + groundMaterial + shadeMaterial + name + simulatedProperties + "}";
        }
    }
}


