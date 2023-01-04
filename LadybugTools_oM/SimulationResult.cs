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
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class SimulationResult : BHoMObject
    {
        [Description("The EPW file associated with this object.")]
        public virtual string EpwFile { get; set; } = string.Empty;
        [Description("The ground material used in the processing of this object.")]
        public virtual ILBTMaterial GroundMaterial { get; set; } = new OpaqueMaterial();
        [Description("The shade material used in the processing of this object.")]
        public virtual ILBTMaterial ShadeMaterial { get; set; } = new OpaqueMaterial();
        [Description("The identifier used to distinguish existing results for this object.")]
        public virtual string Identifier { get; set; } = string.Empty;

        // simulated properties
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedAboveTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedBelowTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedDiffuseRadiation { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedDirectRadiation { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedLongwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedShortwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedTotalRadiation { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedAboveTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedBelowTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDiffuseRadiation { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDirectRadiation { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedLongwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedShortwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedTotalRadiation { get; set; } = new CustomObject();
    }
}


