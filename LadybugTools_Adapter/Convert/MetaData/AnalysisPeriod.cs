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
                StartMonth = (int)customObject["st_month"],
                StartDay = (int)customObject["st_day"],
                StartHour = (int)customObject["st_hour"],
                EndMonth = (int)customObject["end_month"],
                EndDay = (int)customObject["end_day"],
                EndHour = (int)customObject["end_hour"],
                IsLeapYear = (bool)customObject["is_leap_year"],
                TimeStep = (int)customObject["timestep"]
            };
        }

        public static Dictionary<string, object> FromAnalysisPeriod(BH.oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            return new Dictionary<string, object>
            {
                { "type", "AnalysisPeriod" },
                { "st_month", analysisPeriod.StartMonth },
                { "st_day", analysisPeriod.StartDay },
                { "st_hour", analysisPeriod.StartHour },
                { "end_month", analysisPeriod.EndMonth },
                { "end_day", analysisPeriod.EndDay },
                { "end_hour", analysisPeriod.EndHour },
                { "is_leap_year", analysisPeriod.IsLeapYear },
                { "timestep", analysisPeriod.TimeStep }
            };
        }
    }
}
