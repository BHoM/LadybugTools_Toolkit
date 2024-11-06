using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Engine.LadyBugTools
{
    public static partial class Query
    {
        public static Dictionary<string, object> EPWKeyInfo(string EpwFilePath, EPWKey epwKey)
        {
            List<string> valuesAsStrings = new List<string>();
            if (!maps.TryGetValue(epwKey, out int keyIndex))
            {
                BH.Engine.Base.Compute.RecordError($"Cannot retrieve information about the provided EPW key {epwKey}");
                return null;
            }

            using (StreamReader file = System.IO.File.OpenText(EpwFilePath))
            {
                int index = 0;
                while (!file.EndOfStream)
                {
                    string line = file.ReadLine();
                    if (index < 8)
                    {
                        index++;
                        continue;
                    }

                    valuesAsStrings.Add(line.Split(',')[keyIndex]);

                    index++;
                }
            }

            double[] values = new double[8760];

            for (int i = 0; i < valuesAsStrings.Count; i++)
                values[i] = double.Parse(valuesAsStrings[i]); //assume that if getting the strings before has worked correctly, then the values can be parsed without error.

            Dictionary<string, object> output = new Dictionary<string, object>();

            output["max"] = values.Max();
            output["min"] = values.Min();
            output["mean"] = values.Average();
            output["median"] = values.OrderBy(x => x).ToList()[4379];
            output["range"] = (double)output["max"] - (double)output["min"];
            output["values"] = values;
            return output;
        }

        private static Dictionary<EPWKey, int> maps = new Dictionary<EPWKey, int>()
        {
            { EPWKey.DryBulbTemperature, 6 },
            { EPWKey.DewPointTemperature, 7 },
            { EPWKey.RelativeHumidity, 8 },
            { EPWKey.AtmosphericStationPressure, 9 },
            { EPWKey.ExtraterrestrialHorizontalRadiation, 10 },
            { EPWKey.ExtraterrestrialDirectNormalRadiation, 11 },
            { EPWKey.HorizontalInfraredRadiationIntensity, 12 },
            { EPWKey.GlobalHorizontalRadiation, 13 },
            { EPWKey.DirectNormalRadiation, 14 },
            { EPWKey.DiffuseHorizontalRadiation, 15 },
            { EPWKey.GlobalHorizontalIlluminance, 16 },
            { EPWKey.DirectNormalIlluminance, 17 },
            { EPWKey.DiffuseHorizontalIlluminance, 18 },
            { EPWKey.ZenithLuminance, 19 },
            { EPWKey.WindDirection, 20 },
            { EPWKey.WindSpeed, 21 },
            { EPWKey.TotalSkyCover, 22 },
            { EPWKey.OpaqueSkyCover, 23 },
            { EPWKey.Visibility, 24 },
            { EPWKey.CeilingHeight, 25 },
            { EPWKey.PrecipitableWater, 26 },
            { EPWKey.AerosolOpticalDepth, 29 },
            { EPWKey.SnowDepth, 30 },
            { EPWKey.DaysSinceLastSnowfall, 31 }
        };
    }
}
