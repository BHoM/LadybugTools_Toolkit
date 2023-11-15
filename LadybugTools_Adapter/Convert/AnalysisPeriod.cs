using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.AnalysisPeriod AnalysisPeriod(BH.Adapter.LadybugTools.AnalysisPeriod oldObject)
        {
            return new oM.LadybugTools.AnalysisPeriod()
            {
                StartMonth = oldObject.StMonth,
                StartDay = oldObject.StDay,
                StartHour = oldObject.StHour,
                EndMonth = oldObject.EndMonth,
                EndDay = oldObject.EndDay,
                EndHour = oldObject.EndHour,
                IsLeapYear = oldObject.IsLeapYear,
                TimeStep = oldObject.Timestep,
            };
        }
    }
}
