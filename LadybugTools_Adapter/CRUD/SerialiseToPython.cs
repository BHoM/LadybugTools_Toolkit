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
        public static bool SerialiseToPython(List<ILadybugTools> objects, ActionConfig actionConfig = null)
        {
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config is null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid LadybugConfig.");
                return false;
            }
            if (objects.Count() == 0)
            {
                BH.Engine.Base.Compute.RecordError("Please put an input into objects.");
                return false;
            }
            foreach (var item in objects)
            {
                bool success = Convert.FromBHoM(item, config.JsonFile);
                if (!success)
                {
                    return false;
                }
            }

            return true;
        }
    }
}
