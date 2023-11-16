using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.AnalysisPeriod ToAnalysisPeriod(Dictionary<string, object> customObject)
        {
            return new oM.LadybugTools.AnalysisPeriod()
            {
                StartMonth = (int)customObject["StMonth"],
                StartDay = (int)customObject["StDay"],
                StartHour = (int)customObject["StHour"],
                EndMonth = (int)customObject["EndMonth"],
                EndDay = (int)customObject["EndDay"],
                EndHour = (int)customObject["EndHour"],
                IsLeapYear = (bool)customObject["IsLeapYear"],
                TimeStep = (int)customObject["TimeStep"]
            };
        }

        public static Dictionary<string, object> FromAnalysisPeriod(BH.oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            return new Dictionary<string, object>
            {
                { "Type", "AnalysisPeriod" },
                { "StMonth", analysisPeriod.StartMonth },
                { "StDay", analysisPeriod.StartDay },
                { "StHour", analysisPeriod.StartHour },
                { "EndMonth", analysisPeriod.EndMonth },
                { "EndDay", analysisPeriod.EndDay },
                { "EndHour", analysisPeriod.EndHour },
                { "IsLeapYear", analysisPeriod.IsLeapYear },
                { "TimeStep", analysisPeriod.TimeStep }
            };
        }
    }
}
