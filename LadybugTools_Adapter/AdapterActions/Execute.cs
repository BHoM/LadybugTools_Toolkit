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

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override Output<List<object>, bool> Execute(IExecuteCommand command, ActionConfig actionConfig = null)
        {
            Output<List<object>, bool> output = new Output<List<object>, bool>() { Item1 = new List<object>(), Item2 = false };

            List<object> temp = IRunCommand(command);

            output.Item1 = temp;
            output.Item2 = !(BH.Engine.Base.Query.CurrentEvents().Where(x => x.Type == oM.Base.Debugging.EventType.Error).Any());

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
            PythonEnvironment env = Engine.LadybugTools.Compute.InstallPythonEnv_LBT(true);
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Materials_{DateTime.Now:yyyyMMdd}.json",
                    Directory = Path.GetTempPath()
                }
            };

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_material.py");

                string cmdCommand = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> materialObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            return materialObjects.Where(m => (m as IEnergyMaterialOpaque).Name.Contains(command.Filter)).ToList();
        }

        /**************************************************/

        private List<object> RunCommand(GetTypologyCommand command)
        {
            PythonEnvironment env = Engine.LadybugTools.Compute.InstallPythonEnv_LBT(true);
            LadybugConfig config = new LadybugConfig()
            {
                JsonFile = new FileSettings()
                {
                    FileName = $"LBTBHoM_Typologies_{DateTime.Now:yyyyMMdd}.json",
                    Directory = Path.GetTempPath()
                }
            };

            if (!File.Exists(config.JsonFile.GetFullFileName()))
            {
                string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "get_typology.py");

                string cmdCommand = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";

                Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);
            }

            List<object> typologyObjects = Pull(new FilterRequest(), actionConfig: config).ToList();

            return typologyObjects.Where(m => (m as Typology).Name.Contains(command.Filter)).ToList();
        }

        /**************************************************/

        private List<object> RunCommand(RunSimulationCommand command)
        {
            // validation prior to passing to Python
            if (command.EpwFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"{nameof(command.EpwFile)} input cannot be null.");
                return null;
            }

            if (!File.Exists(command.EpwFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"{command.EpwFile.GetFullFileName()} does not exist.");
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
                EpwFile = Path.GetFullPath(command.EpwFile.GetFullFileName()).Replace(@"\", "/"),
                GroundMaterial = command.GroundMaterial,
                ShadeMaterial = command.ShadeMaterial,
                Name = Engine.LadybugTools.Compute.SimulationID(command.EpwFile.GetFullFileName(), command.GroundMaterial, command.ShadeMaterial)
            };

            // push object to json file
            Push(new List<SimulationResult>() { simulationResult }, actionConfig: config);

            // locate the Python executable and file containing the simulation code
            PythonEnvironment env = Engine.LadybugTools.Compute.InstallPythonEnv_LBT(true);
            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "simulation_result.py");

            // run the simulation
            string cmdCommand = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> simulationResultPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

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

            // locate the Python executable and file containing the simulation code
            PythonEnvironment env = Engine.LadybugTools.Compute.InstallPythonEnv_LBT(true);
            string script = Path.Combine(Engine.Python.Query.DirectoryCode(), "LadybugTools_Toolkit\\src\\ladybugtools_toolkit\\bhom\\wrapped", "external_comfort.py");

            // run the calculation
            string cmdCommand = $"{env.Executable} {script} -j \"{config.JsonFile.GetFullFileName()}\"";
            Engine.Python.Compute.RunCommandStdout(command: cmdCommand, hideWindows: true);

            // reload from Python results
            List<object> externalComfortPopulated = Pull(new FilterRequest(), actionConfig: config).ToList();

            // remove temporary file
            File.Delete(config.JsonFile.GetFullFileName());

            return externalComfortPopulated;
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
