/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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

using BH.oM.LadybugTools;
using LadybugTools_oM.Enums;
using BH.oM.Reflection.Attributes;

using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Data.SQLite;
using System.Reflection;
using System.Linq;
using System;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Create an EnergyPlusResult object from a SQLite file containing Ladybug HourlyContinuousCollections.")]
        [Input("sqliteFile", "A SQLite containing Ladybug HourlyContinuousCollections.")]
        [Input("energyPlusResultType", "A result attribute to query from the SQLite simulation results.")]
        [Output("energyPlusResult", "An EnergyPlusResult object.")]
        public static List<LBDateTime> DateTimes(this AnalysisPeriod analysisPeriod)
        {
            BH.Engine.Reflection.Compute.RecordWarning("This method doesn't work for any non-continuous analysis periods and as such needs reworking to account for these!");
            // TODO - Fix method to work for ANY AnalysisPeriod

            List<LBDateTime> lbDateTimes = new List<LBDateTime>();

            DateTime currentTime = analysisPeriod.StartDateTime();
            DateTime endTime = analysisPeriod.EndDateTime();

            while (currentTime < endTime)
            {
                lbDateTimes.Add(currentTime.ToLBDateTime());
                currentTime = currentTime.AddHours(1);
            }
                
            return lbDateTimes;
        }

        private static int Year(this AnalysisPeriod analysisPeriod)
        {
            return analysisPeriod.IsLeapYear ? 2020 : 2021;
        }

        private static DateTime StartDateTime(this AnalysisPeriod analysisPeriod)
        {
            return new DateTime(analysisPeriod.Year(), analysisPeriod.StartMonth, analysisPeriod.StartDay, analysisPeriod.StartHour, 0, 0);
        }

        private static DateTime EndDateTime(this AnalysisPeriod analysisPeriod)
        {
            return new DateTime(analysisPeriod.Year(), analysisPeriod.EndMonth, analysisPeriod.EndDay, analysisPeriod.EndHour, 0, 0);
        }

        private static LBDateTime ToLBDateTime(this DateTime datetime)
        {
            return new LBDateTime()
            {
                LeapYear = DateTime.IsLeapYear(datetime.Year),
                Month = datetime.Month,
                Day = datetime.Day,
                Hour = datetime.Hour,
                Minute = datetime.Minute
            };
        }
    }
}