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

using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class UTCIData : ISimulationData
    {
        [Description("The ratio of comfortable hours / total hours.")]
        public virtual double ComfortableRatio { get; set; } = double.NaN;

        [Description("The ratio of heat stress hours / total hours.")]
        public virtual double HotRatio { get; set; } = double.NaN;

        [Description("The ratio of cold stress hours / total hours.")]
        public virtual double ColdRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime active comfortable hours / daytime active hours.")]
        public virtual double DaytimeComfortableRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime active heat stress hours / daytime active hours.")]
        public virtual double DaytimeHotRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime active cold stress hours / daytime active hours.")]
        public virtual double DaytimeColdRatio { get; set; } = double.NaN;
    }
}
