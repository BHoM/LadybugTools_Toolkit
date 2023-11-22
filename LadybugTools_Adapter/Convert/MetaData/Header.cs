using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        //TODO - find out if Header is needed or should just return an AnalysisPeriod
        public static BH.oM.LadybugTools.Header ToHeader(Dictionary<string, object> oldObject)
        {
            if (oldObject["data_type"].GetType() == typeof(CustomObject))
                oldObject["data_type"] = (oldObject["data_type"] as CustomObject).CustomData;

            if (oldObject["analysis_period"].GetType() == typeof(CustomObject))
                oldObject["analysis_period"] = (oldObject["analysis_period"] as CustomObject).CustomData;

            if (oldObject["metadata"].GetType() == typeof(CustomObject))
                oldObject["metadata"] = (oldObject["metadata"] as CustomObject).CustomData;

            return new oM.LadybugTools.Header()
            {
                Unit = (string)oldObject["unit"],
                DataType = ToDataType(oldObject["data_type"] as Dictionary<string, object>),
                AnalysisPeriod = ToAnalysisPeriod(oldObject["analysis_period"] as Dictionary<string, object>),
                Metadata = (Dictionary<string, object>)oldObject["metadata"],
            };
        }

        public static Dictionary<string, object> FromHeader(BH.oM.LadybugTools.Header header)
        {
            return new Dictionary<string, object>()
            {
                { "type", "Header" },
                { "data_type", FromDataType(header.DataType) },
                { "unit", header.Unit },
                { "analysis_period", FromAnalysisPeriod(header.AnalysisPeriod) },
                { "metadata", header.Metadata }
            };
        }
    }
}
