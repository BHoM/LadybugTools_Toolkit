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
                if (oldObject["shelters"].GetType() == typeof(CustomObject))
                    oldObject["shelters"] = (List<Dictionary<string, object>>)(oldObject["shelters"] as CustomObject).CustomData["values"];

                foreach (Dictionary<string, object> shelter in oldObject["shelters"] as List<Dictionary<string, object>>)
                    shelters.Add(ToShelter(shelter));
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
                    if (double.TryParse(value.ToString(), out double result))
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
                RadiantTemperatureAdjustment = radiantTemperatureAdjustment
            };
        }

        public static string FromTypology(oM.LadybugTools.Typology typology)
        {
            string identifier = typology.Name;
            string shelters = "[" + string.Join(", ", typology.Shelters.Select(s => FromShelter(s)).ToList()) + "]";
            string evaporativeCoolingEffect = "[" + string.Join(", ", typology.EvaporativeCoolingEffect) + "]";
            string targetWindSpeed = "[" + string.Join(", ", typology.TargetWindSpeed) + "]";
            string radiantTemperatureAdjustment = "[" + string.Join(", ", typology.RadiantTemperatureAdjustment) + "]";
            return @"{""type"": ""Typology""," + 
                @"""identifier"": """ + identifier + 
                @""", ""shelters"": " + shelters + 
                @", ""evaporative_cooling_effect"": " + evaporativeCoolingEffect + 
                @", ""target_wind_speed"": " + targetWindSpeed + 
                @", ""radiant_temperature_adjustment"": " + radiantTemperatureAdjustment + "}";
        }
    }
}
