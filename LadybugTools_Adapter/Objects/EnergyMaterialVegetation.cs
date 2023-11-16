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


using System.ComponentModel;
using BH.oM.Base;
using BH.oM.Base.Attributes;

namespace BH.Adapter.LadybugTools
{
    [NoAutoConstructor]
    public class EnergyMaterialVegetation : BHoMObject, IEnergyMaterialOpaque, ILBTSerialisable
    {
        public virtual string Type { get; set; } = "EnergyMaterialVegetation";

        public virtual string Identifier { get; set; }

        public virtual double Thickness { get; set; }

        public virtual double Conductivity { get; set; }

        public virtual double Density { get; set; }

        public virtual double SpecificHeat { get; set; }

        public virtual oM.LadybugTools.Roughness Roughness { get; set; }

        public virtual double SoilThermalAbsorptance { get; set; }

        public virtual double SoilSolarAbsorptance { get; set; }

        public virtual double SoilVisibleAbsorptance { get; set; }

        public virtual double PlantHeight { get; set; }

        public virtual double LeafAreaIndex { get; set; }

        public virtual double LeafReflectivity { get; set; }

        public virtual double LeafEmissivity { get; set; }

        public virtual double MinStomatalResist { get; set; }
    }
}
