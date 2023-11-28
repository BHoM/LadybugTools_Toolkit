﻿using BH.oM.Base;
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

            return new ExternalComfort()
            {
                SimulationResult = simulationResult,
                Typology = typology,
                DryBulbTemperature = simulatedProperties[0],
                RelativeHumidity = simulatedProperties[1],
                WindSpeed = simulatedProperties[2],
                MeanRadiantTemperature = simulatedProperties[3],
                UniversalThermalClimateIndex = simulatedProperties[4]
            };
        }

        public static string FromExternalComfort(ExternalComfort externalComfort)
        {
            string type = "\"type\": \"ExternalComfort\", ";
            string simulationResult = "\"simulation_result\": " + FromSimulationResult(externalComfort.SimulationResult) + ", ";
            string typology = "\"typology\": " + FromTypology(externalComfort.Typology);
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
