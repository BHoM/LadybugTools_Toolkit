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
            string BaseUnit;
            if (oldObject.ContainsKey("base_unit"))
            {
                BaseUnit = (string)oldObject["base_unit"];
            }
            else
            {
                BaseUnit = "";
            }
            return new oM.LadybugTools.DataType()
            {
                Data_Type = (string)oldObject["data_type"],
                Name = (string)oldObject["name"],
                BaseUnit = BaseUnit
            };
        }

        public static Dictionary<string, object> FromDataType(BH.oM.LadybugTools.DataType dataType)
        {
            Dictionary<string, object> returnDict = new Dictionary<string, object>
            {
                { "type", "GenericDataType" },
                { "name", dataType.Name },
                { "data_type", dataType.Data_Type }
            };
            if (dataType.BaseUnit != "")
            {
                returnDict.Add("base_unit", dataType.BaseUnit);
            }
            return returnDict;
        }
    }
}
