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
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class AnalysisPeriod : ILadybugTools
    {
        [Description("The Ladybug datatype of this object, used for deserialisation.")]
        public virtual string Type { get; set; } = "AnalysisPeriod";

        [Description("The start month.")]
        public virtual int StMonth { get; set; }

        [Description("The start day.")]
        public virtual int StDay { get; set; }

        [Description("The start hour.")]
        public virtual int StHour { get; set; }

        [Description("The end month.")]
        public virtual int EndMonth { get; set; }

        [Description("The end day.")]
        public virtual int EndDay { get; set; }

        [Description("The end hour.")]
        public virtual int EndHour { get; set; }

        [Description("Boolean flag for whether this represents a leap year.")]
        public virtual bool IsLeapYear { get; set; }

        [Description("The number of timesteps per hour.")]
        public virtual int Timestep { get; set; }
    }
}
