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
    public class ExternalComfort : BHoMObject
    {
        [Description("The SimulationResult associated with this object.")]
        public virtual SimulationResult SimulationResult { get; set; } = new SimulationResult();
        [Description("The typology in the processing of this object.")]
        public virtual Typology Typology { get; set; } = new Typology();

        // simulated properties

        [Description("The calculated property from this object.")]
        public virtual CustomObject DryBulbTemperature { get; set; } = new CustomObject();
        [Description("The calculated property from this object.")]
        public virtual CustomObject RelativeHumidity { get; set; } = new CustomObject();
        [Description("The calculated property from this object.")]
        public virtual CustomObject WindSpeed { get; set; } = new CustomObject();
        [Description("The calculated property from this object.")]
        public virtual CustomObject MeanRadiantTemperature { get; set; } = new CustomObject();
        [Description("The calculated property from this object.")]
        public virtual CustomObject UniversalThermalClimateIndex { get; set; } = new CustomObject();
    }
}

