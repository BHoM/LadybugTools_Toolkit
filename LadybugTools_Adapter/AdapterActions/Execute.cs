using BH.oM.Adapter;
using BH.oM.Adapter.Commands;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override Output<List<object>, bool> Execute(IExecuteCommand command, ActionConfig actionConfig = null)
        {
            Output<List<object>, bool> output = new Output<List<object>, bool>() { Item1 = new List<object>(), Item2 = false };

            if (command == null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid ILadybugCommand to execute.");
                return output;
            }

            ILadybugCommand ladybugCommand = command as ILadybugCommand;
            if (ladybugCommand == null)
            {
                BH.Engine.Base.Compute.RecordError($"The command {command.GetType().FullName} is not valid for the LadybugTools Adapter. Please use a LadybugCommand, or use the correct adapter for the input command.");
                return output;
            }

            if (actionConfig == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid LadybugConfig to use with this adapter.");
                return output;
            }
            
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config == null)
            {
                BH.Engine.Base.Compute.RecordError($"The ActionConfig provided: {actionConfig.GetType().FullName} is not valid for the LadybugTools Adapter. Please use a valid LadybugConfig.");
                return output;
            }

            if (config.JsonFile == null)
            {
                BH.Engine.Base.Compute.RecordError($"Please provide a valid JsonFile FileSettings object in the LadybugConfig.");
                return output;
            }


            return new Output<List<object>, bool>();
        }
    }
}
