using BH.Engine.Adapter;
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override List<object> Push(IEnumerable<object> objects, string tag = "", PushType pushType = PushType.AdapterDefault, ActionConfig actionConfig = null)
        {
            if (actionConfig is null)
            {
                BH.Engine.Base.Compute.RecordError("Please input an actionconfig before setting active to true.");
                return new List<object>();
            }
            
            SerialiseToPython(objects.Cast<ILadybugTools>().ToList(), actionConfig);
            return objects.ToList();
        }
    }
}
