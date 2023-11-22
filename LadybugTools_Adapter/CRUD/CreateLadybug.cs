using BH.oM.Adapter;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;


namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public static bool CreateLadybug(List<ILadybugTools> objects, ActionConfig actionConfig = null)
        {
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config is null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid LadybugConfig.");
                return false;
            }

            if (objects.Count() == 0)
            {
                BH.Engine.Base.Compute.RecordError("Please input an object.");
                return false;
            }
            
            if (objects.Count > 1)
                BH.Engine.Base.Compute.RecordWarning("The LadybugToolsAdapter does not currently support pushing multiple objects to one file, only the first object will be saved.");

            if (!Convert.FromBHoM(objects[0], config.JsonFile))
            {
                BH.Engine.Base.Compute.RecordError("An error occurred during conversion to json:");
                return false;
            }
            return true;
        }
    }
}
