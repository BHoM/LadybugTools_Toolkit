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
        public virtual ILadybugToolsMaterial GroundMaterial { get; set; } = new OpaqueMaterial();
        [Description("The shade material used in the processing of this object.")]
        public virtual ILadybugToolsMaterial ShadeMaterial { get; set; } = new OpaqueMaterial();
        [Description("The identifier used to distinguish existing results for this object.")]
        public virtual string Identifier { get; set; } = string.Empty;

        // simulated properties
        [Description("The simulated property from this object.")]
<<<<<<< HEAD
        public virtual CustomObject ShadedDownDirectIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedDownDiffuseIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedDownTotalIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedDownTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedUpDiffuseIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedUpDirectIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedUpTotalIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedUpTemperature { get; set; } = new CustomObject();
=======
        public virtual CustomObject ShadedDownTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedUpTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedLongwaveMeanRadiantTemperatureDelta { get; set; } = new CustomObject();
>>>>>>> 19bbfa46f3a8c77a9a97c818aff2b6653a5166aa
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedShortwaveMeanRadiantTemperatureDelta { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
<<<<<<< HEAD
        public virtual CustomObject ShadedShortwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject ShadedMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDownDiffuseIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDownDirectIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDownTotalIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDownTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedUpDiffuseIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedUpDirectIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedUpTotalIrradiance { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedUpTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedLongwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedShortwaveMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
=======
        public virtual CustomObject ShadedMeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedDownTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedUpTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedRadiantTemperature { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedLongwaveMeanRadiantTemperatureDelta { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
        public virtual CustomObject UnshadedShortwaveMeanRadiantTemperatureDelta { get; set; } = new CustomObject();
        [Description("The simulated property from this object.")]
>>>>>>> 19bbfa46f3a8c77a9a97c818aff2b6653a5166aa
        public virtual CustomObject UnshadedMeanRadiantTemperature { get; set; } = new CustomObject();
    }
}


