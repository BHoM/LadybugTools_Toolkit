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
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class Typology : BHoMObject, ILadybugTools
    {
        [Description("The name of this Typology.")]
        public override string Name { get; set; } = string.Empty;

        [Description("The shelters for this Typology.")]
        public virtual List<Shelter> Shelters { get; set; } = new List<Shelter>();
        
        [Description("The proportion of evaporative cooling to add to this Typology.")]
        public virtual List<double> EvaporativeCoolingEffect { get; set; } = Enumerable.Repeat<double>(0.0, 8760).ToList();
        
        [Description("A multiplier to apply to the wind speed component of this Typology.")]
        public virtual List<double?> TargetWindSpeed { get; set; } = Enumerable.Repeat<double?>(null, 8760).ToList();
        
        [Description("A reduction or increase in MRT to be applied to results generated using this Typology.")]
        public virtual List<double> RadiantTemperatureAdjustment { get; set; } = Enumerable.Repeat<double>(0.0, 8760).ToList();
    }
}
