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

using BH.Engine.Python;
using BH.oM.Base;
using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using BH.oM.Python;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Get a predefined ExternalComfortTypology by it's name.")]
        [Input("typologyName", "The name of a pre-defined typology.")]
        [Output("ExternalComfortTypology", "An ExternalComfortTypology object.")]
        public static ExternalComfortTypology ExternalComfortTypology(string typology)
        {
            BH.oM.Python.PythonEnvironment env = Compute.LadybugToolsToolkitPythonEnvironment(true);

            // get a list of typologies that have been predefined in the Python code, as a custom object for each
            string pythonScript = string.Join("\n", new List<string>()
            {
                "from ladybugtools_toolkit.external_comfort import Typologies",
                "from ladybugtools_toolkit.external_comfort.encoder.encoder import Encoder",
                "import json",
                "",
                "d = {}",
                "for typology in Typologies:",
                "    d[typology.name] = typology.value.to_dict()",
                "print(json.dumps(d, cls = Encoder))",
            });

            string output = env.RunPythonString(pythonScript).Trim();
            CustomObject typologies = Serialiser.Convert.FromJson(output) as CustomObject;
            List<string> typologyIds = new List<string>();
            foreach (string typologyName in typologies.CustomData.Keys)
            {
                typologyIds.Add(typologyName);
            }

            if (!typologyIds.Contains(typology))
            {
                BH.Engine.Base.Compute.RecordError($"The typology given is not predefined in the Python source code. Please use one of [\n{String.Join(",\n    ", typologyIds)}\n].");
            }

            // create the ECTypology from the given string name of the Typology
            CustomObject predefinedTypology = (typologies.CustomData[typology] as CustomObject);

            List<ExternalComfortShelter> shelters = new List<ExternalComfortShelter>();
            foreach (CustomObject shelterObj in ((List<System.Object>)predefinedTypology.CustomData["shelters"]).Cast<CustomObject>())
            {
                ExternalComfortShelter shelter = new ExternalComfortShelter();
                shelter.Porosity = (double)shelterObj.CustomData["porosity"];

                List<double> shelterAzimuthRange = new List<double>();
                foreach (double shelterObjAz in ((List<System.Object>)shelterObj.CustomData["azimuth_range"]).Cast<double>())
                {
                    shelterAzimuthRange.Add(shelterObjAz);
                }
                shelter.StartAzimuth = shelterAzimuthRange[0];
                shelter.EndAzimuth = shelterAzimuthRange[1];

                List<double> shelterAltitudeRange = new List<double>();
                foreach (double shelterObjAlt in ((List<System.Object>)shelterObj.CustomData["altitude_range"]).Cast<double>())
                {
                    shelterAltitudeRange.Add(shelterObjAlt);
                }
                shelter.StartAltitude = shelterAltitudeRange[0];
                shelter.EndAltitude = shelterAltitudeRange[1];

                shelters.Add(shelter);
            }

            BH.oM.LadybugTools.ExternalComfortTypology ecTypology = new ExternalComfortTypology() {
                Name = (string)predefinedTypology.CustomData["name"],
                EvaporativeCoolingEffectiveness = (double)predefinedTypology.CustomData["evaporative_cooling_effectiveness"],
                Shelters = shelters
            };

            // return the typology
            return ecTypology;
        }
    }
}

