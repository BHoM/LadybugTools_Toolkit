using BH.oM.Adapter;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override IEnumerable<object> Pull(IRequest request, PullType pullType = PullType.AdapterDefault, ActionConfig actionConfig = null) 
        {
            if (request != null)
            {
                FilterRequest filterRequest = request as FilterRequest;
                return Read(filterRequest.Type, actionConfig: actionConfig);
            }
            else
                return new List<object>();
        }
    }
}
