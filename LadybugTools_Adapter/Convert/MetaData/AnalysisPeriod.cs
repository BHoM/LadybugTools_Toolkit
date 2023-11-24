/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.AnalysisPeriod ToAnalysisPeriod(Dictionary<string, object> oldObject)
        {
            int startMonth = 1;
            int startDay = 1;
            int startHour = 0;
            int endMonth = 12;
            int endDay = 31;
            int endHour = 23;
            int timeStep = 1;
            bool isLeapYear = false;

            try
            {
                startMonth = (int)oldObject["st_month"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the start month of the AnalysisPeriod. returning start month as default ({startMonth}).\n The error: {ex}");
            }

            try
            {
                startDay = (int)oldObject["st_day"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the start day of the AnalysisPeriod. returning start day as default ({startDay}).\n The error: {ex}");
            }

            try
            {
                startHour = (int)oldObject["st_hour"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the start hour of the AnalysisPeriod. returning start hour as default ({startHour}).\n The error: {ex}");
            }

            try
            {
                endMonth = (int)oldObject["end_month"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the end month of the AnalysisPeriod. returning end month as default ({endMonth}).\n The error: {ex}");
            }

            try
            {
                endDay = (int)oldObject["end_day"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the end day of the AnalysisPeriod. returning end day as default ({endDay}).\n The error: {ex}");
            }

            try
            {
                endHour = (int)oldObject["end_hour"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the end hour of the AnalysisPeriod. returning end hour as default ({endHour}).\n The error: {ex}");
            }

            try
            {
                isLeapYear = (bool)oldObject["is_leap_year"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when trying to determine whether the AnalysisPeriod is a leap year. returning IsLeapYear as default (false).\n The error: {ex}");
            }

            try
            {
                timeStep = (int)oldObject["timestep"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the time step of the AnalysisPeriod. returning time step as default ({timeStep}).\n The error: {ex}");
            }

            return new oM.LadybugTools.AnalysisPeriod()
            {
                StartMonth = startMonth,
                StartDay = startDay,
                StartHour = startHour,
                EndMonth = endMonth,
                EndDay = endDay,
                EndHour = endHour,
                IsLeapYear = isLeapYear,
                TimeStep = timeStep
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
