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
        [DisplayText("EPW File")]
        [Description("The EPW file associated with this object.")]
        public virtual FileSettings EpwFile { get; set; }

        [DisplayText("Ground Material")]
        [Description("The ground material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; }

        [DisplayText("Shade Material")]
        [Description("The shade material used in the processing of this object.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; }

        [DisplayText("Name")]
        [Description("The identifier used to distinguish existing results for this object.")]
        public override string Name { get; set; }

        // simulated properties

        [DisplayText("Shaded Down Temperature")]
        [Description("The Shaded Down Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedDownTemperature { get; } = null;

        [DisplayText("Shaded Up Temperature")]
        [Description("The Shaded Up Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedUpTemperature { get; } = null;

        [DisplayText("Shaded Radiant Temperature")]
        [Description("The Shaded Radiant Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedRadiantTemperature { get; } = null;

        [DisplayText("Shaded Longwave Mean Radiant Temperature Delta")]
        [Description("The Shaded Longwave Mean Radiant Temperature Delta used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedLongwaveMeanRadiantTemperatureDelta { get; } = null;

        [DisplayText("Shaded Shortwave Mean Radiant Temperature Delta")]
        [Description("The Shaded Shortwave Mean Radiant Temperature Delta used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedShortwaveMeanRadiantTemperatureDelta { get; } = null;

        [DisplayText("Shaded Mean Radiant Temperature")]
        [Description("The Shaded Mean Radiant Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection ShadedMeanRadiantTemperature { get; } = null;

        [DisplayText("Unshaded Down Temperature")]
        [Description("The Unshaded Down Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection UnshadedDownTemperature { get; } = null;

        [DisplayText("Unshaded Up Temperature")]
        [Description("The Unshaded Up Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection UnshadedUpTemperature { get; } = null;

        [DisplayText("Unshaded Radiant Temperature")]
        [Description("The Unshaded Radiant Temperature used in the processing of this object")]
        public virtual HourlyContinuousCollection UnshadedRadiantTemperature { get; } = null;

        [DisplayText("Unshaded Longwave Mean Radiant Temperature Delta")]
        [Description("The Unshaded Longwave Mean Radiant Temperature Delta used in the processing of this object")]
        public virtual HourlyContinuousCollection UnshadedLongwaveMeanRadiantTemperatureDelta { get; } = null;

        [DisplayText("Unshaded Shortwave Mean Radiant Temperature Delta")]
        [Description("The Unshaded Shortwave Mean Radiant Temperature Delta used in the processing of this object")]
        public virtual HourlyContinuousCollection UnshadedShortwaveMeanRadiantTemperatureDelta { get; } = null;

        [DisplayText("Unshaded Mean Radiant Temperature")]
        [Description("The Unshaded Mean Radiant Temperature used in the processing of this object")]
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


