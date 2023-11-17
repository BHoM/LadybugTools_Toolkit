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
            var obj = Engine.Serialiser.Convert.FromJson(json);
            Dictionary<string, object> LBTObject = null;
            if (obj.GetType() == typeof(CustomObject))
            {
                LBTObject = (obj as CustomObject).CustomData;
            }
            else
            {
                BH.Engine.Base.Compute.RecordWarning("The object was not deserialised as a CustomObject, are you sure that this file came from a LadybugTools Python object? \n Trying to cast to Dictionary...");
                LBTObject = obj as Dictionary<string, object>;
            }
            switch (LBTObject["type"] as string)
            {
                case "AnalysisPeriod":
                    return ToAnalysisPeriod(LBTObject);
                case "DataType":
                    return ToDataType(LBTObject);
                case "EnergyMaterial":
                    return ToEnergyMaterial(LBTObject);
                case "EnergyMaterialVegetation":
                    return ToEnergyMaterialVegetation(LBTObject);
                case "EPW":
                    return ToEPW(LBTObject);
                case "Header":
                    return ToHeader(LBTObject);
                case "HourlyContinuousCollection":
                    return ToHourlyContinuousCollection(LBTObject);
                case "Location":
                    return ToLocation(LBTObject);
                default:
                    BH.Engine.Base.Compute.RecordError("The json file given is not convertable to a LadybugTools object.");
                    return null;
            }
        }
    }
}