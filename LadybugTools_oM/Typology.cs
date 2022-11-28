/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class Typology : BHoMObject
    {

        [Description("The name of this Typology.")]
        public override string Name { get; set; } = "Openfield";
        [Description("The shelters for this ExternalComfortTypology.")]
        public virtual List<Shelter> Shelters { get; set; } = new List<Shelter>();
        [Description("The proportion of evaporative cooling to add to this ExternalComfortTypology.")]
        public virtual double EvaporativeCoolingEffectiveness { get; set; } = 0;
        [Description("A factor to apply to the wind speed component of this ExternalComfortTypology.")]
        public virtual double WindSpeedAdjustment { get; set; } = 0;
    }
}

