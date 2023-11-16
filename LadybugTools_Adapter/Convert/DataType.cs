using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.DataType DataType(BH.Adapter.LadybugTools.DataType oldObject)
        {
            return new oM.LadybugTools.DataType()
            {
                Data_Type = oldObject.Data_Type
            };
        }
    }
}
