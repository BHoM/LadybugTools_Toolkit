using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.HourlyContinuousCollection HourlyContinuousCollection(BH.Adapter.LadybugTools.HourlyContinuousCollection oldObject)
        {
            return new oM.LadybugTools.HourlyContinuousCollection()
            {
                Values = oldObject.Values,
            };
        }
    }
}
