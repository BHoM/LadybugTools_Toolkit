using BH.oM.LadybugTools;
using System;
using BH.Engine.Serialiser;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using BH.oM.Adapter;
using System.IO;
using BH.Engine.Adapter;
using BH.oM.Base;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ILadybugTools ToBHoM(this FileSettings jsonFile)
        {
            string json = File.ReadAllText(jsonFile.GetFullFileName());
            Dictionary<string, object> LBTObject = Engine.Serialiser.Convert.FromJson(json) as Dictionary<string, object>;
            switch (LBTObject["type"] as string)
            {
                case "AnalysisPeriod":
                    return ToAnalysisPeriod(LBTObject);
                case "DataType":
                    return ToDataType(LBTObject);
                case "HourlyContinuousCollection":
                    return ToHourlyContinuousCollection(LBTObject);
                default:
                    BH.Engine.Base.Compute.RecordError("The json file given is not convertable to a LadybugTools object.");
                    return null;
            }
        }
    }
}