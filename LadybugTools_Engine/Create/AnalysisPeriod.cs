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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create an AnalysisPeriod object.")]
        [Input("startMonth", "The starting month of the analysis period. 1-12.")]
        [Input("startDay", "The starting day of the analysis period. 1-31.")]
        [Input("startHour", "The starting hour of the analysis period. 0-23.")]
        [Input("endMonth", "The ending month of the analysis period. 1-12.")]
        [Input("endDay", "The ending day of the analysis period. 1-31.")]
        [Input("endHour", "The ending hour of the analysis period. 0-23.")]
        [Input("isLeapYear", "Whether the analysis period represents a leap year.")]
        [Input("timestep", "The number of timesteps per hour. One of {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60}.")]
        [Output("analysisPeriod", "An AnalysisPeriod object.")]
        public static AnalysisPeriod AnalysisPeriod(int startMonth = 1, int startDay = 1, int startHour = 0, int endMonth = 12, int endDay = 31, int endHour = 23, bool isLeapYear = false, int timestep = 1)
        {
            if (startMonth < 1 || startMonth > 12 || endMonth < 1 || endMonth > 12)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(startMonth)} and {nameof(endMonth)} must be between 1 and 12.");
                return null;
            }

            if (startDay < 1 || startDay > 31 || endDay < 1 || endDay > 31)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(startDay)} and {nameof(endDay)} must be between 1 and 31.");
                return null;
            }

            if (startHour < 0 || startHour > 23 || endHour < 0 || endHour > 23)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(startHour)} and {nameof(endHour)} must be between 0 and 23.");
                return null;
            }

            List<int> allowedTimesteps = new List<int>() { 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 };
            if (!allowedTimesteps.Contains(timestep))
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(timestep)} must be one of {string.Join(", ", allowedTimesteps)}.");
                return null;
            }

            return new AnalysisPeriod()
            {
                StartMonth = startMonth,
                StartDay = startDay,
                StartHour = startHour,
                EndMonth = endMonth,
                EndDay = endDay,
                EndHour = endHour,
                IsLeapYear = isLeapYear,
                TimeStep = timestep
            };
        }
    }
}
