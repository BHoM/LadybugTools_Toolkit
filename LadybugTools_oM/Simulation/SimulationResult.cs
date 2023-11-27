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

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class SimulationResult : BHoMObject, ILadybugTools
    {
        [Description("The EPW file associated with this object.")]
        public virtual string EpwFile { get; set; }

        [Description("The ground material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; }

        [Description("The shade material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; }

        [Description("The identifier used to distinguish existing results for this object.")]
        public override string Name { get; set; }

        // simulated properties

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedDownTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedUpTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedRadiantTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedLongwaveMeanRadiantTemperatureDelta { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedShortwaveMeanRadiantTemperatureDelta { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedMeanRadiantTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedDownTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedUpTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedRadiantTemperature { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedLongwaveMeanRadiantTemperatureDelta { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedShortwaveMeanRadiantTemperatureDelta { get; set; }

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedMeanRadiantTemperature { get; set; }
    }
}
