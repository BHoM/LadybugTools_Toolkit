﻿/*
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

using BH.oM.Base;
using BH.oM.Geometry;
using LadybugTools_oM.Enums;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class Header : BHoMObject
    {
        [Description("Data type.")]
        public virtual DataType DataType { get; set; } = DataType.Undefined;
        
        [Description("Unit.")]
        public virtual string UnitType { get; set; } = "";

        [Description("Analysis period.")]
        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        [Description("Optional meta-data to be associated with the Header.")]
        public virtual Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
    }
}
