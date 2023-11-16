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
            return new oM.LadybugTools.Header()
            {
                Unit = (string)oldObject["unit"],
                DataType = ToDataType(oldObject["data_type"] as Dictionary<string, object>),
                AnalysisPeriod = ToAnalysisPeriod(oldObject["analysis_period"] as Dictionary<string, object>),
                Metadata = (Dictionary<string, string>)oldObject["metadata"],
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
                { "MetaData", header.Metadata }
            };
        }
    }
}
