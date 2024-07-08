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
        public static ISimulationData IToSimulationData(this CustomObject oldObject, ISimulationData toUpdate)
        {
            return ToSimulationData(oldObject.CustomData, toUpdate as dynamic);
        }

        private static CollectionData ToSimulationData(this Dictionary<string, object> oldData, CollectionData toUpdate)
        {
            try
            {
                toUpdate.Description = (oldData["description"] as List<object>).Select(x => x.ToString()).ToList();
            }
            catch { }

            double.TryParse(oldData["highest_value"].ToString(), out double result);
            toUpdate.HighestValue = result;

            double.TryParse(oldData["lowest_value"].ToString(), out result);
            toUpdate.LowestValue = result;

            DateTime.TryParse(oldData["highest_time"].ToString(), out DateTime date);
            toUpdate.HighestTime = date;

            DateTime.TryParse(oldData["lowest_time"].ToString(), out date);
            toUpdate.LowestTime = date;

            double.TryParse(oldData["highest_average_month_value"].ToString(), out result);
            toUpdate.HighestAverageMonthValue = result;

            int.TryParse(oldData["highest_average_month"].ToString(), out int month);
            toUpdate.HighestAverageMonth = month;

            double.TryParse(oldData["lowest_average_month_value"].ToString(), out result);
            toUpdate.LowestAverageMonthValue = result;

            int.TryParse(oldData["lowest_average_month"].ToString(), out month);
            toUpdate.LowestAverageMonth = month;

            return toUpdate;
        }

        private static WindroseData ToSimulationData(this Dictionary<string, object> oldData, WindroseData toUpdate)
        {
            try
            {
                toUpdate.Description = (oldData["description"] as List<object>).Select(x => x.ToString()).ToList();
            }
            catch { }

            double.TryParse(oldData["prevailing_direction"].ToString(), out double result);
            toUpdate.PrevailingDirection = result;

            double.TryParse(oldData["prevailing_speed"].ToString(), out result);
            toUpdate.PrevailingSpeed = result;


            DateTime.TryParse(oldData["prevailing_time"].ToString(), out DateTime date);
            toUpdate.PrevailingTime = date;

            double.TryParse(oldData["max_speed"].ToString(), out result);
            toUpdate.MaxSpeed = result;

            double.TryParse(oldData["max_speed_direction"].ToString(), out result);
            toUpdate.MaxSpeedDirection = result;

            DateTime.TryParse(oldData["max_speed_time"].ToString(), out date);
            toUpdate.MaxSpeedTime = date;

            double.TryParse(oldData["calm_count"].ToString(), out result);
            toUpdate.NumberOfCalmHours = result;

            double.TryParse(oldData["calm_percent"].ToString(), out result);
            toUpdate.PercentageOfCalmHours = result;

            return toUpdate;
        }

        private static SunPathData ToSimulationData(this Dictionary<string, object> oldData, SunPathData toUpdate)
        {
            return toUpdate;
        }
    }
}