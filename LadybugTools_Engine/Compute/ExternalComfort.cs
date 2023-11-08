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
using System.IO;
using System.Linq;
using System;

using BH.Engine.Geometry;
using BH.Engine.Serialiser;
using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using BH.oM.Python;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Run an External Comfort simulation and return results.")]
        [Input("simulationResult", "A simulation result object.")]
        [Input("typology", "An ExternalComfortTypology.")]
        [Output("externalComfort", "An external comfort result object containing simulation results.")]
        public static ExternalComfort ExternalComfort(SimulationResult simulationResult, Typology typology)
        {
            if (simulationResult == null)
            {
                BH.Engine.Base.Compute.RecordError("simulationResult input cannot be null.");
                return null;
            }

            if (typology == null)
            {
                BH.Engine.Base.Compute.RecordError("typology input cannot be null.");
                return null;
            }

            if (typology.EvaporativeCoolingEffect.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError("Typology evaporative cooling effect must be a list of values 8760 long.");
                return null;
            }
            foreach (double evapClg in typology.EvaporativeCoolingEffect)
            {
                if (evapClg < 0 || evapClg > 1)
                {
                    BH.Engine.Base.Compute.RecordError("Typology evaporative cooling effect must be between 0 and 1.");
                    return null;
                }
            }

            if (typology.RadiantTemperatureAdjustment.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError("Typology radiant temperature adjustment must be a list of values 8760 long.");
                return null;
            }

            if (typology.TargetWindSpeed.Count() != 8760)
            {
                BH.Engine.Base.Compute.RecordError("Typology target wind speed must be a list of values 8760 long.");
                return null;
            }
            foreach (double? tgtWindSpd in typology.TargetWindSpeed)
            {
                if (tgtWindSpd == null)
                {
                    continue;
                }
                else
                {
                    if (tgtWindSpd < 0)
                    {
                        BH.Engine.Base.Compute.RecordError("Typology target wind speed must be greater than or eqaul to 0 (or null).");
                        return null;
                    }
                }
            }

            foreach (Shelter shelter in typology.Shelters)
            {
                if (!BH.Engine.Geometry.Create.Polyline(shelter.Vertices).IsPlanar())
                {
                    BH.Engine.Base.Compute.RecordError("A shelter in this Typology is not planar.");
                    return null;
                }
                if (shelter.WindPorosity.Count() != 8760)
                {
                    BH.Engine.Base.Compute.RecordError("Shelter wind porosity must be a list of values 8760 long.");
                    return null;
                }
                foreach (double windPorosity in shelter.WindPorosity)
                {
                    if (windPorosity < 0 || windPorosity > 1)
                    {
                        BH.Engine.Base.Compute.RecordError("Shelter wind porosity must be between 0 and 1.");
                        return null;
                    }
                }
                if (shelter.RadiationPorosity.Count() != 8760)
                {
                    BH.Engine.Base.Compute.RecordError("Shelter radiation porosity must be a list of values 8760 long.");
                    return null;
                }
                foreach (double radiationPorosity in shelter.RadiationPorosity)
                {
                    if (radiationPorosity < 0 || radiationPorosity > 1)
                    {
                        BH.Engine.Base.Compute.RecordError("Shelter radiation porosity must be between 0 and 1.");
                        return null;
                    }
                }
            }

            if (typology.Identifier == "")
            {
                dynamic wsAvg;
                if (typology.TargetWindSpeed.Where(x => x.HasValue).Count() == 0)
                {
                    wsAvg = "EPW";
                }
                else
                {
                    wsAvg = typology.TargetWindSpeed.Where(x => x.HasValue).Average(x => x.Value);
                }
                if (typology.Shelters.Count() == 0)
                {
                    typology.Identifier = $"ec{typology.EvaporativeCoolingEffect.Average()}_ws{wsAvg}_mrt{typology.RadiantTemperatureAdjustment.Average()}";
                }
                else
                {
                    typology.Identifier = $"shelters{typology.Shelters.Count()}_ec{typology.EvaporativeCoolingEffect.Average()}_ws{wsAvg}_mrt{typology.RadiantTemperatureAdjustment.Average()}";
                }
                Base.Compute.RecordNote($"This typology has been automatically named \"{typology.Identifier}\". This can be overriden with the 'identifier' parameter of Typology.");
            }

            // construct the base object
            ExternalComfort externalComfort = new ExternalComfort()
            {
                SimulationResult = simulationResult,
                Typology = typology,
            };
            string jsonPreSimulation = externalComfort.ToJson();
            string jsonFile = Path.Combine(Path.GetTempPath(), $"LBTBHoM_{Guid.NewGuid()}.json");
            File.WriteAllText(jsonFile, jsonPreSimulation);

            // locate the Python executable and file containing the simulation code
            PythonEnvironment env = InstallPythonEnv_LBT(true);
            string script = Path.Combine(Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "external_comfort.py");

            // run the calculation
            string command = $"{env.Executable} {script} -j \"{jsonFile}\"";
            Python.Compute.RunCommandStdout(command: command, hideWindows: true);

            // reload from Python results
            string jsonPostSimulation = File.ReadAllText(jsonFile);
            BH.oM.LadybugTools.ExternalComfort externalComfortPopulated = (BH.oM.LadybugTools.ExternalComfort)BH.Engine.Serialiser.Convert.FromJson(jsonPostSimulation);

            // remove temporary file
            File.Delete(jsonFile);

            return externalComfortPopulated;
        }
    }
}
