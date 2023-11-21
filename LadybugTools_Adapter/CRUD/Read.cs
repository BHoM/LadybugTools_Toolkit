using BH.Engine.Adapter;
using BH.Engine.Serialiser;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        protected override IEnumerable<IBHoMObject> IRead(Type type, IList indices = null, ActionConfig actionConfig = null)
        {
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config == null)
            {
                BH.Engine.Base.Compute.RecordError($"The type of actionConfig provided: {actionConfig.GetType().FullName} is not valid for this adapter. Please provide a valid LadybugConfig actionConfig.");
                return new List<IBHoMObject>();
            }

            if (config.JsonFile == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid JsonFile FileSettings object.");
                return new List<IBHoMObject>();
            }

            if (!System.IO.File.Exists(config.JsonFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"The file at {config.JsonFile.GetFullFileName()} does not exist to pull from.");
                return new List<IBHoMObject>();
            }

            List<IBHoMObject> rtnObjs = new List<IBHoMObject>();

            rtnObjs.Add(config.JsonFile.ToBHoM());

            return rtnObjs;

        }
    }
}
