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


using BH.oM.Base;
using BH.oM.Base.Attributes;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class ExternalComfort : BHoMObject, ILadybugTools
    {
        [Description("The SimulationResult associated with this object.")]
        public virtual SimulationResult SimulationResult { get; set; }

        [Description("The typology in the processing of this object.")]
        public virtual Typology Typology { get; set; }

        // simulated properties

        [Description("The calculated property from this object.")]
        public virtual HourlyContinuousCollection DryBulbTemperature { get; set; }

        [Description("The calculated property from this object.")]
        public virtual HourlyContinuousCollection RelativeHumidity { get; set; }

        [Description("The calculated property from this object.")]
        public virtual HourlyContinuousCollection WindSpeed { get; set; }

        [Description("The calculated property from this object.")]
        public virtual HourlyContinuousCollection MeanRadiantTemperature { get; set; }

        [Description("The calculated property from this object.")]
        public virtual HourlyContinuousCollection UniversalThermalClimateIndex { get; set; }
    }
}


