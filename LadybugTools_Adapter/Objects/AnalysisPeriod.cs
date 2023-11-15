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


using BH.oM.Base;
using BH.oM.Base.Attributes;
using System.ComponentModel;

namespace BH.Adapter.LadybugTools
{
    [NoAutoConstructor]
    public class AnalysisPeriod : BHoMObject, ILBTSerialisable
    {
        public virtual string Type { get; set; } = "AnalysisPeriod";

        public virtual int StMonth { get; set; }

        public virtual int StDay { get; set; }

        public virtual int StHour { get; set; }

        public virtual int EndMonth { get; set; }

        public virtual int EndDay { get; set; }

        public virtual int EndHour { get; set; }

        public virtual bool IsLeapYear { get; set; }

        public virtual int Timestep { get; set; }
    }
}
