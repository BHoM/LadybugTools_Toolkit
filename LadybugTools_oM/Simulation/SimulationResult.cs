/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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


using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.Base.Attributes;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class SimulationResult : BHoMObject, ILadybugTools, IImmutable
    {
        [Description("The EPW file associated with this object.")]
        public virtual FileSettings EpwFile { get; set; }

        [Description("The ground material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; }

        [Description("The shade material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; }

        [Description("The identifier used to distinguish existing results for this object.")]
        public override string Name { get; set; }

        // simulated properties

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedDownTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedUpTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedRadiantTemperature { get; } = null;     

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedLongwaveMeanRadiantTemperatureDelta { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedShortwaveMeanRadiantTemperatureDelta { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection ShadedMeanRadiantTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedDownTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedUpTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedRadiantTemperature { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedLongwaveMeanRadiantTemperatureDelta { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedShortwaveMeanRadiantTemperatureDelta { get; } = null;

        [Description("The simulated property from this object.")]
        public virtual HourlyContinuousCollection UnshadedMeanRadiantTemperature { get; } = null;

        public SimulationResult(FileSettings epwFile = null, IEnergyMaterialOpaque groundMaterial = null, IEnergyMaterialOpaque shadeMaterial = null, string name = null, HourlyContinuousCollection shadedDownTemperature = null, HourlyContinuousCollection shadedUpTemperature = null, HourlyContinuousCollection shadedRadiantTemperature = null, HourlyContinuousCollection shadedLongwaveMeanRadiantTemperatureDelta = null, HourlyContinuousCollection shadedShortwaveMeanRadiantTemperatureDelta = null, HourlyContinuousCollection shadedMeanRadiantTemperature = null, HourlyContinuousCollection unshadedDownTemperature = null, HourlyContinuousCollection unshadedUpTemperature = null, HourlyContinuousCollection unshadedRadiantTemperature = null, HourlyContinuousCollection unshadedLongwaveMeanRadiantTemperatureDelta = null, HourlyContinuousCollection unshadedShortwaveMeanRadiantTemperatureDelta = null, HourlyContinuousCollection unshadedMeanRadiantTemperature = null)
        {
            EpwFile = epwFile;
            GroundMaterial = groundMaterial;
            ShadeMaterial = shadeMaterial;
            Name = name;
            ShadedDownTemperature = shadedDownTemperature;
            ShadedUpTemperature = shadedUpTemperature;
            ShadedRadiantTemperature = shadedRadiantTemperature;
            ShadedLongwaveMeanRadiantTemperatureDelta = shadedLongwaveMeanRadiantTemperatureDelta;
            ShadedShortwaveMeanRadiantTemperatureDelta = shadedShortwaveMeanRadiantTemperatureDelta;
            ShadedMeanRadiantTemperature = shadedMeanRadiantTemperature;
            UnshadedDownTemperature = unshadedDownTemperature;
            UnshadedUpTemperature = unshadedUpTemperature;
            UnshadedRadiantTemperature = unshadedRadiantTemperature;
            UnshadedLongwaveMeanRadiantTemperatureDelta = unshadedLongwaveMeanRadiantTemperatureDelta;
            UnshadedShortwaveMeanRadiantTemperatureDelta = unshadedShortwaveMeanRadiantTemperatureDelta;
            UnshadedMeanRadiantTemperature = unshadedMeanRadiantTemperature;
        }
    }
}


