using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {

        /**************************************************/
        /* Public methods - Interface                     */
        /**************************************************/

        public List<object> IRunCommand(IExecuteCommand command, ActionConfig actionConfig)
        {
            if (command == null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid Ladybug Command to execute.");
                return new List<object>();
            }

            return RunCommand(command as dynamic, actionConfig);
        }

        /**************************************************/
        /* Private methods - Fallback                     */
        /**************************************************/

        private List<object> RunCommand(IExecuteCommand command, ActionConfig actionConfig)
        {
            BH.Engine.Base.Compute.RecordError($"The command {command.GetType().FullName} is not valid for the LadybugTools Adapter. Please use a LadybugCommand, or use the correct adapter for the input command.");
            return new List<object>();
        }
    }
}
