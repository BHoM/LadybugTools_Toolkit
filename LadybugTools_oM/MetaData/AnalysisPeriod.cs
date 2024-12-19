/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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
using BH.oM.Base.Attributes;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class AnalysisPeriod : BHoMObject, ILadybugTools
    {
        [Description("The start month.")]
        public virtual int StartMonth { get; set; } = 1;

        [Description("The start day.")]
        public virtual int StartDay { get; set; } = 1;

        [Description("The start hour.")]
        public virtual int StartHour { get; set; } = 0;

        [Description("The end month.")]
        public virtual int EndMonth { get; set; } = 12;

        [Description("The end day.")]
        public virtual int EndDay { get; set; } = 31;

        [Description("The end hour.")]
        public virtual int EndHour { get; set; } = 23;

        [Description("Boolean flag for whether this represents a leap year.")]
        public virtual bool IsLeapYear { get; set; } = false;

        [Description("The number of timesteps per hour.")]
        public virtual int TimeStep { get; set; } = 1;
    }
}


