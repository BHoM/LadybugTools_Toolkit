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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System.ComponentModel;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Create an ID for a simulation.")]
        [Input("epwFile", "An EPW file path.")]
        [Input("groundMaterial", "A material object.")]
        [Input("shadeMaterial", "A material object.")]
        [Output("id", "A simulation ID.")]
        public static string SimulationId(string epwFile, ILadybugToolsMaterial groundMaterial, ILadybugToolsMaterial shadeMaterial)
        {
            if (epwFile == null)
            {
                BH.Engine.Base.Compute.RecordError("epwFile input cannot be null.");
                return null;
            }

            if (groundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError("groundMaterial input cannot be null.");
                return null;
            }

            if (shadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError("shadeMaterial input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(epwFile))
            {
                BH.Engine.Base.Compute.RecordError($"{epwFile} doesn't appear to exist!");
                return null;
            }

            string epwId = Convert.SanitiseString(Path.GetFileNameWithoutExtension(epwFile));
            string groundMaterialId = Convert.SanitiseString(groundMaterial.Identifier);
            string shadeMaterialId = Convert.SanitiseString(shadeMaterial.Identifier);
            return $"{epwId}__{groundMaterialId}__{shadeMaterialId}";
        }
    }
}

