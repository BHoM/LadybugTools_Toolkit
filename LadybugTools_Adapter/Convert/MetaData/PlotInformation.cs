/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
using System.Security.Principal;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {

        public static PlotInformation ToPlotInformation(this CustomObject oldObject, ISimulationData toUpdate)
        {
            PlotInformation plotInformation = new PlotInformation();

            plotInformation.Image = oldObject.CustomData["figure"].ToString();

            plotInformation.OtherData = IToSimulationData((oldObject.CustomData["data"] as CustomObject).CustomData, toUpdate);

            return plotInformation;
 
        }

        private static ISimulationData IToSimulationData(Dictionary<string, object> oldObject, ISimulationData toUpdate)
        {
            return ToSimulationData(oldObject, toUpdate as dynamic);
        }

        private static CollectionData ToSimulationData(this Dictionary<string, object> oldData, CollectionData toUpdate)
        {

            if (!double.TryParse(oldData["highest"].ToString(), out double result))
                result = double.NaN;

            toUpdate.HighestValue = result;

            if (!double.TryParse(oldData["lowest"].ToString(), out result))
                result = double.NaN;

            toUpdate.LowestValue = result;

            if (!DateTime.TryParse(oldData["highest_index"].ToString(), out DateTime date))
                date = DateTime.MinValue;

            toUpdate.HighestIndex = date;

            if (!DateTime.TryParse(oldData["lowest_index"].ToString(), out date))
                date = DateTime.MinValue;

            toUpdate.LowestIndex = date;

            if (!double.TryParse(oldData["median"].ToString(), out result))
                result = double.NaN;

            toUpdate.MedianValue = result;

            if (!double.TryParse(oldData["mean"].ToString(), out result))
                result = double.NaN;

            toUpdate.MeanValue = result;

            try
            {
                List<object> means = oldData["month_means"] as List<object>;

                int monthIndex = 0;

                foreach (object mean in means)
                {
                    if (!double.TryParse(mean.ToString(), out double value))
                        value = double.NaN;

                    toUpdate.MonthlyMeans[monthIndex] = value;
                    monthIndex++;
                }
            }
            catch { }

            try
            {
                List<List<object>> ranges = oldData["month_diurnal_ranges"] as List<List<object>>;
                List<List<double>> monthlyRanges = new List<List<double>>();

                int monthIndex = 0;

                foreach (List<object> range in ranges)
                {
                    List<double> values = new List<double>();

                    foreach (object value in range)
                    {
                        if (!double.TryParse(value.ToString(), out result))
                            result = double.NaN;

                        values.Add(result);
                    }

                    toUpdate.MonthlyDiurnalRanges[monthIndex] = values;
                    monthIndex++;
                }
            }
            catch { }


            return toUpdate;
        }

        private static WindroseData ToSimulationData(this Dictionary<string, object> oldData, WindroseData toUpdate)
        {
            if (!double.TryParse(oldData["prevailing_95percentile"].ToString(), out double result))
                result = double.NaN;
            toUpdate.PrevailingPercentile95 = result;

            try
            {
                List<object> tuple = oldData["prevailing_direction"] as List<object>;
                int index = 0;

                foreach (object value in tuple)
                {
                    if (!double.TryParse(value.ToString(), out result))
                        result = double.NaN;

                    toUpdate.PrevailingDirection[index] = result;
                    index++;
                }
            }
            catch { }

            if (!double.TryParse(oldData["prevailing_50percentile"].ToString(), out result))
                result = double.NaN;
            toUpdate.PrevailingPercentile50 = result;

            if (!double.TryParse(oldData["95percentile"].ToString(), out result))
                result = double.NaN;
            toUpdate.Percentile95 = result;

            if (!double.TryParse(oldData["50percentile"].ToString(), out result))
                result = double.NaN;
            toUpdate.Percentile50 = result;

            if (!double.TryParse(oldData["calm_percent"].ToString(), out result))
                result = double.NaN;
            toUpdate.PercentageOfCalmHours = result;

            return toUpdate;
        }

        private static SunPathData ToSimulationData(this Dictionary<string, object> oldData, SunPathData toUpdate)
        {
            try
            {
                Dictionary<string, object> december_object = (oldData["december_solstice"] as CustomObject).CustomData;

                Dictionary<string, object> sunset = (december_object["sunset"] as CustomObject).CustomData;

                if (!double.TryParse(sunset["azimuth"].ToString(), out double result))
                    result = double.NaN;
                toUpdate.DecemberSolstice.SunsetAzimuth = result;

                if (!DateTime.TryParse(sunset["time"].ToString(), out DateTime date))
                    date = DateTime.MinValue;
                toUpdate.DecemberSolstice.SunsetTime = date;

                Dictionary<string, object> sunrise = (december_object["sunrise"] as CustomObject).CustomData;

                if (!double.TryParse(sunrise["azimuth"].ToString(), out result))
                    result = double.NaN;
                toUpdate.DecemberSolstice.SunriseAzimuth = result;

                if (!DateTime.TryParse(sunrise["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.DecemberSolstice.SunriseTime = date;

                Dictionary<string, object> noon = (december_object["noon"] as CustomObject).CustomData;

                if (!double.TryParse(noon["altitude"].ToString(), out result))
                    result = double.NaN;
                toUpdate.DecemberSolstice.NoonAltitude = result;

                if (!DateTime.TryParse(noon["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.DecemberSolstice.NoonTime = date;
            }
            catch { }

            try
            {
                Dictionary<string, object> march_object = (oldData["march_equinox"] as CustomObject).CustomData;

                Dictionary<string, object> sunset = (march_object["sunset"] as CustomObject).CustomData;

                if (!double.TryParse(sunset["azimuth"].ToString(), out double result))
                    result = double.NaN;
                toUpdate.MarchEquinox.SunsetAzimuth = result;

                if (!DateTime.TryParse(sunset["time"].ToString(), out DateTime date))
                    date = DateTime.MinValue;
                toUpdate.MarchEquinox.SunsetTime = date;

                Dictionary<string, object> sunrise = (march_object["sunrise"] as CustomObject).CustomData;

                if (!double.TryParse(sunrise["azimuth"].ToString(), out result))
                    result = double.NaN;
                toUpdate.MarchEquinox.SunriseAzimuth = result;

                if (!DateTime.TryParse(sunrise["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.MarchEquinox.SunriseTime = date;

                Dictionary<string, object> noon = (march_object["noon"] as CustomObject).CustomData;

                if (!double.TryParse(noon["altitude"].ToString(), out result))
                    result = double.NaN;
                toUpdate.MarchEquinox.NoonAltitude = result;

                if (!DateTime.TryParse(noon["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.MarchEquinox.NoonTime = date;
            }
            catch { }

            try
            {
                Dictionary<string, object> june_object = (oldData["june_solstice"] as CustomObject).CustomData;

                Dictionary<string, object> sunset = (june_object["sunset"] as CustomObject).CustomData;

                if (!double.TryParse(sunset["azimuth"].ToString(), out double result))
                    result = double.NaN;
                toUpdate.JuneSolstice.SunsetAzimuth = result;

                if (!DateTime.TryParse(sunset["time"].ToString(), out DateTime date))
                    date = DateTime.MinValue;
                toUpdate.JuneSolstice.SunsetTime = date;

                Dictionary<string, object> sunrise = (june_object["sunrise"] as CustomObject).CustomData;

                if (!double.TryParse(sunrise["azimuth"].ToString(), out result))
                    result = double.NaN;
                toUpdate.JuneSolstice.SunriseAzimuth = result;

                if (!DateTime.TryParse(sunrise["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.JuneSolstice.SunriseTime = date;

                Dictionary<string, object> noon = (june_object["noon"] as CustomObject).CustomData;

                if (!double.TryParse(noon["altitude"].ToString(), out result))
                    result = double.NaN;
                toUpdate.JuneSolstice.NoonAltitude = result;

                if (!DateTime.TryParse(noon["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.JuneSolstice.NoonTime = date;
            }
            catch { }

            try
            {
                Dictionary<string, object> september_object = (oldData["september_equinox"] as CustomObject).CustomData;

                Dictionary<string, object> sunset = (september_object["sunset"] as CustomObject).CustomData;

                if (!double.TryParse(sunset["azimuth"].ToString(), out double result))
                    result = double.NaN;
                toUpdate.SeptemberEquinox.SunsetAzimuth = result;

                if (!DateTime.TryParse(sunset["time"].ToString(), out DateTime date))
                    date = DateTime.MinValue;
                toUpdate.SeptemberEquinox.SunsetTime = date;

                Dictionary<string, object> sunrise = (september_object["sunrise"] as CustomObject).CustomData;

                if (!double.TryParse(sunrise["azimuth"].ToString(), out result))
                    result = double.NaN;
                toUpdate.SeptemberEquinox.SunriseAzimuth = result;

                if (!DateTime.TryParse(sunrise["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.SeptemberEquinox.SunriseTime = date;

                Dictionary<string, object> noon = (september_object["noon"] as CustomObject).CustomData;

                if (!double.TryParse(noon["altitude"].ToString(), out result))
                    result = double.NaN;
                toUpdate.SeptemberEquinox.NoonAltitude = result;

                if (!DateTime.TryParse(noon["time"].ToString(), out date))
                    date = DateTime.MinValue;
                toUpdate.SeptemberEquinox.NoonTime = date;
            }
            catch { }

            return toUpdate;
        }

        private static UTCIData ToSimulationData(this Dictionary<string, object> oldData, UTCIData toUpdate)
        {
            if (!double.TryParse(oldData["comfortable_ratio"].ToString(), out double result))
                result = double.NaN;
            toUpdate.ComfortableRatio = result;

            if (!double.TryParse(oldData["hot_ratio"].ToString(), out result))
                result = double.NaN;
            toUpdate.HotRatio = result;

            if (!double.TryParse(oldData["cold_ratio"].ToString(), out result))
                result = double.NaN;
            toUpdate.ColdRatio = result;

            if (!double.TryParse(oldData["daytime_comfortable"].ToString(), out result))
                result = double.NaN;
            toUpdate.DaytimeComfortableRatio = result;

            if (!double.TryParse(oldData["daytime_hot"].ToString(), out result))
                result = double.NaN;
            toUpdate.DaytimeHotRatio = result;

            if (!double.TryParse(oldData["daytime_cold"].ToString(), out result))
                result = double.NaN;
            toUpdate.DaytimeColdRatio = result;

            return toUpdate;
        }
    }
}