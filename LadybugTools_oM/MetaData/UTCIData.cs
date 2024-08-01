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
        [Description("The ratio of comfortable hours to total hours. Comfortable hours are hours between 9 and 26°C.")]
        public virtual double ComfortableRatio { get; set; } = double.NaN;

        [Description("The ratio of heat stress hours to total hours. Heat stress hours are hours greater than 26°C.")]
        public virtual double HeatStressRatio { get; set; } = double.NaN;

        [Description("The ratio of cold stress hours to total hours. Cold stress hours are hours less than 9°C.")]
        public virtual double ColdStressRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime comfortable hours to daytime hours. Daytime comfortable hours are hours between 9 and 26°C and between 07:00-22:59.")]
        public virtual double DaytimeComfortableRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime heat stress hours to daytime hours. Daytime heat stress hours are hours greater than 26°C and between 07:00-22:59.")]
        public virtual double DaytimeHeatStressRatio { get; set; } = double.NaN;

        [Description("The ratio of daytime cold stress hours to daytime hours. Daytime cold stress hours are hours less than 9°C and between 07:00-22:59.")]
        public virtual double DaytimeColdStressRatio { get; set; } = double.NaN;
    }
}
