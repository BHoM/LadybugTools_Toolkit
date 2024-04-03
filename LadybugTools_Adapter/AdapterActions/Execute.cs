/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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

using BH.Engine.Adapter;
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.Adapter.Commands;
using BH.oM.Base;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using BH.oM.Python;
using BH.Engine.Python;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using BH.Engine.Base;
using System.Drawing;
using BH.Engine.Serialiser;
using BH.Engine.LadyBugTools;
using System.Reflection;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        bool m_executeSuccess = false;
        public override Output<List<object>, bool> Execute(IExecuteCommand command, ActionConfig actionConfig = null)
        {
            m_executeSuccess = false;
            Output<List<object>, bool> output = new Output<List<object>, bool>() { Item1 = new List<object>(), Item2 = false };

            List<object> temp = IRunCommand(command);

            output.Item1 = temp;
            output.Item2 = m_executeSuccess;

            return output;
        }

        /**************************************************/
        /* Public methods - Interface                     */
        /**************************************************/

        public List<object> IRunCommand(IExecuteCommand command)
        {
            if (command == null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid Ladybug Command to execute.");
                return new List<object>();
            }

            return RunCommand(command as dynamic);
        }

        /**************************************************/
        /* Private methods - Run Ladybug Command          */
        /**************************************************/

        private List<object> RunCommand(GetMaterialCommand command)
        {
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Materials.json",
                    Directory = Path.GetTempPath()
                }
            };

            TimeSpan timeSinceLastUpdate = DateTime.Now - File.GetCreationTime(config.JsonFile.GetFullFileName());
            if (timeSinceLastUpdate.Days > 30)
                File.Delete(config.JsonFile.GetFullFileName());

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_material.py");

                string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> materialObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            m_executeSuccess = true;
            return materialObjects.Where(m => (m as IEnergyMaterialOpaque).Name.Contains(command.Filter)).ToList();
        }

        /**************************************************/

        private List<object> RunCommand(GetTypologyCommand command)
        {
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Typologies.json",
                    Directory = Path.GetTempPath()
                }
            };

            TimeSpan timeSinceLastUpdate = DateTime.Now - File.GetCreationTime(config.JsonFile.GetFullFileName());
            if (timeSinceLastUpdate.Days > 30)
                File.Delete(config.JsonFile.GetFullFileName());

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_typology.py");

                string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> typologyObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            m_executeSuccess = true;
            return typologyObjects.Where(m => (m as Typology).Name.Contains(command.Filter)).ToList();
        }

        /**************************************************/

        private List<object> RunCommand(RunSimulationCommand command)
        {
            // validation prior to passing to Python
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"FIle '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.GroundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.GroundMaterial)} input cannot be null.");
                return null;
            }

            if (command.ShadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.ShadeMaterial)} input cannot be null.");
                return null;
            }

            // construct adapter and config
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_{Guid.NewGuid()}.json",
                    Directory = Path.GetTempPath()
                }
            };

            // construct the base object and file to be passed to Python for simulation
            SimulationResult simulationResult = new SimulationResult()
            {
                EpwFile = Path.GetFullPath(command.EPWFile.GetFullFileName()).Replace(@"\", "/"),
                GroundMaterial = command.GroundMaterial,
                ShadeMaterial = command.ShadeMaterial,
                Name = Engine.LadybugTools.Compute.SimulationID(command.EPWFile.GetFullFileName(), command.GroundMaterial, command.ShadeMaterial)
            };

            // push object to json file
            Push(new List<SimulationResult>() { simulationResult }, actionConfig: config);

            // locate the Python file containing the simulation code
            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "simulation_result.py");

            // run the simulation
            string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> simulationResultPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            m_executeSuccess = true;
            return simulationResultPopulated;
        }

        /**************************************************/

        private List<object> RunCommand(RunExternalComfortCommand command)
        {
            if (command.SimulationResult == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.SimulationResult)} input cannot be null.");
                return null;
            }

            if (command.Typology == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.Typology)} input cannot be null.");
                return null;
            }

            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_{Guid.NewGuid()}.json",
                    Directory = Path.GetTempPath()
                }
            };

            // construct the base object
            ExternalComfort externalComfort = new ExternalComfort()
            {
                SimulationResult = command.SimulationResult,
                Typology = command.Typology,
            };

            // push objects to json file
            Push(new List<ExternalComfort>() { externalComfort }, actionConfig: config);

            // locate the Python file containing the simulation code
            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "external_comfort.py");

            // run the calculation
            string cmdCommand = $"{m_environment.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> externalComfortPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            m_executeSuccess = true;
            return externalComfortPopulated;
        }

        /**************************************************/

        private List<object> RunCommand(HeatPlotCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile}' does not exist.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "heatmap.py");

            //check if the colourmap is valid for user warning, but run with input anyway as the map could be defined separately.
            string colourMap = command.ColourMap;
            if (colourMap.ColourMapValidity())
                colourMap = colourMap.ToColourMap().FromColourMap();

            // run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -dtk \"{command.EPWKey.ToText()}\" -cmap \"{colourMap}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result };
        }

        /**************************************************/

        private List<object> RunCommand(WindroseCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.AnalysisPeriod == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid analysis period to run this command.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "windrose.py");

            //check if the colourmap is valid for user warning, but run with input anyway as the map could be defined separately.
            string colourMap = command.ColourMap;
            if (colourMap.ColourMapValidity())
                colourMap = colourMap.ToColourMap().FromColourMap();

            // run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -cmap \"{colourMap}\" -bins \"{command.NumberOfDirectionBins}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object> { result };
        }

        /**************************************************/

        private List<object> RunCommand(UTCIHeatPlotCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.GroundMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid ground material to run this command.");
                return null;
            }

            if (command.ShadeMaterial == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid shade material to run this command.");
                return null;
            }

            if (command.Typology == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please input a valid Typology to run this command.");
            }

            if (!(command.BinColours.Count == 10 || command.BinColours.Count == 0))
            {
                BH.Engine.Base.Compute.RecordError($"When overriding bin colours 10 colours must be provided, but {command.BinColours.Count} colours were provided instead.");
                return null;
            }
            List<string> colours = command.BinColours.Select(x => x.ToHexCode()).ToList();

            string hexColours = $"[\"{string.Join("\",\"", colours)}\"]";
            if (hexColours == "[\"\"]")
                hexColours = "[]";

            Dictionary<string, string> inputObjects = new Dictionary<string, string>()
            {
                { "ground_material",  command.GroundMaterial.FromBHoM() },
                { "shade_material", command.ShadeMaterial.FromBHoM() },
                { "typology", command.Typology.FromBHoM() },
                { "bin_colours", hexColours }
            };

            string argFile = Path.GetTempFileName();
            File.WriteAllText(argFile, inputObjects.ToJson());

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "utci_heatmap.py");

            // run the process
            string cmdCommand = $"{m_environment.Executable} \"{script}\" -e \"{epwFile}\" -in \"{argFile}\" -ws \"{command.WindSpeedMultiplier}\" -sp \"{command.OutputLocation}\"";
            string result = "";

            try
            {
                result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError(ex, "An error occurred while running some python.");
            }
            finally
            {
                File.Delete(argFile);
            }

            return new List<object> { result.Split('\n').Last() };
        }

        private List<object> RunCommand(DiurnalPlotCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile}' does not exist.");
                return null;
            }

            if (command.Period == DiurnalPeriod.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid diurnal period.");
                return null;
            }

            if (command.EPWKey == EPWKey.Undefined)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid EPW key.");
                return null;
            }

            command.Title = command.Title.SanitiseString();

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "diurnal.py");

            // run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -dtk \"{command.EPWKey.ToText()}\" -c \"{command.Colour.ToHexCode()}\" -t \"{command.Title}\" -ap \"{command.Period.ToString().ToLower()}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result };
        }

        private List<object> RunCommand(SunPathPlotCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }
            
            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.AnalysisPeriod == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (command.SunSize < 1)
            {
                BH.Engine.Base.Compute.RecordError($"Sun size must be more than or equal to 1.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "sunpath.py");

            //run the process
            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -s {command.SunSize} -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result };
        }

        private List<object> RunCommand(SolarTiltOrientationCommand command)
        {
            if (command.EPWFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EPWFile)} input cannot be null.");
                return null;
            }

            if (!System.IO.File.Exists(command.EPWFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"File '{command.EPWFile.GetFullFileName()}' does not exist.");
                return null;
            }

            if (command.Azimuths < 1)
            {
                BH.Engine.Base.Compute.RecordError($"Azimuths must be more than or equal to 1.");
                return null;
            }

            if (command.Altitudes < 1)
            {
                BH.Engine.Base.Compute.RecordError($"Altitudes must be more than or equal to 1");
                return null;
            }

            if (command.GroundReflectance < 0 || command.GroundReflectance > 1)
            {
                BH.Engine.Base.Compute.RecordError($"Ground reflectance must be between 0 and 1");
                return null;
            }

            if (command.IrradianceType == IrradianceType.Undefined)
            {
                BH.Engine.Base.Compute.RecordError($"Please provide a valid Irradiance Type.");
                return null;
            }

            if (command.AnalysisPeriod == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(AnalysisPeriod)} input cannot be null.");
                return null;
            }

            string epwFile = System.IO.Path.GetFullPath(command.EPWFile.GetFullFileName());

            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped\\plot", "directional_solar_radiation.py");

            string cmdCommand = $"{m_environment.Executable} {script} -e \"{epwFile}\" -az {command.Azimuths} -al {command.Altitudes} -gr {command.GroundReflectance} -ir {command.IrradianceType} -iso {command.Isotropic} -t {command.Title} -ap \"{command.AnalysisPeriod.FromBHoM().Replace("\"", "\\\"")}\" -p \"{command.OutputLocation}\"";
            string result = Engine.Python.Compute.RunCommandStdout(cmdCommand, hideWindows: true);

            m_executeSuccess = true;
            return new List<object>() { result };
        }

        /**************************************************/
        /* Private methods - Fallback                     */
        /**************************************************/

        private List<object> RunCommand(IExecuteCommand command)
        {
            BH.Engine.Base.Compute.RecordError($"The command {command.GetType().FullName} is not valid for the LadybugTools Adapter. Please use a LadybugCommand, or use the correct adapter for the input command.");
            return new List<object>();
        }
    }
}

