using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.DataType ToDataType(Dictionary<string, object> oldObject)
        {
            return new oM.LadybugTools.DataType()
            {
                Data_Type = (string)oldObject["data_type"],
                Name = (string)oldObject["name"]
            };
        }

        public static Dictionary<string, object> FromDataType(BH.oM.LadybugTools.DataType dataType)
        {
            return new Dictionary<string, object>
            {
                { "type", "DataType" },
                { "name", "" },
                { "data_type", dataType.Data_Type }
            };
        }
    }
}
