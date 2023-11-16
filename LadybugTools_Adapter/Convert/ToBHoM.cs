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
            BH.Adapter.LadybugTools.AnalysisPeriod testinput = new AnalysisPeriod();
            CustomObject LBTObject = Engine.Serialiser.Convert.FromJson(json) as CustomObject;
            switch (LBTObject.CustomData["Type"] as string)
            {
                case "AnalysisPeriod":
                    return ToAnalysisPeriod(LBTObject.CustomData);
                case "DataType":
                    return ToDataType(LBTObject.CustomData);
                default:
                    BH.Engine.Base.Compute.RecordError("The json file given is not convertable to a LadybugTools object.");
                    return null;
            }
        }
    }
}