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
            if (actionConfig == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide config settings to pull from a ladybug json file.");
                return new List<IBHoMObject>();
            }
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid LadybugConfig for pulling from ladybug json");
                return new List<IBHoMObject>();
            }
            else if (config.JsonFile == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid JsonFile FileSettings object.");
            }
            if (type == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide the type of object represented in the ladybug json file.");
            }

            List<IBHoMObject> rtnObjs = new List<IBHoMObject>();

            if (type == typeof(AnalysisPeriod))
            {
                rtnObjs.Add(config.JsonFile.ToBHoM());
                return rtnObjs;
            }

            return rtnObjs;
        }
    }
}
