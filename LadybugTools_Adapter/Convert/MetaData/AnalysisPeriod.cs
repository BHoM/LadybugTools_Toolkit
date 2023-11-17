using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.AnalysisPeriod ToAnalysisPeriod(Dictionary<string, object> oldObject)
        {
            return new oM.LadybugTools.AnalysisPeriod()
            {
                StartMonth = (int)oldObject["st_month"],
                StartDay = (int)oldObject["st_day"],
                StartHour = (int)oldObject["st_hour"],
                EndMonth = (int)oldObject["end_month"],
                EndDay = (int)oldObject["end_day"],
                EndHour = (int)oldObject["end_hour"],
                IsLeapYear = (bool)oldObject["is_leap_year"],
                TimeStep = (int)oldObject["timestep"]
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
