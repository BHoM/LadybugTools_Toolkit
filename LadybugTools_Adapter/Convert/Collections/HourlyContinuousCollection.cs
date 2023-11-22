using BH.Engine.Base;
using BH.Engine.Serialiser;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.HourlyContinuousCollection ToHourlyContinuousCollection(Dictionary<string, object> oldObject)
        {
            if (oldObject["header"].GetType() == typeof(CustomObject))
                oldObject["header"] = (oldObject["header"] as CustomObject).CustomData;

            List<object> objectList = oldObject["values"] as List<object>;
            List<ICollection> hourlyValues = new List<ICollection>();

            if (double.TryParse(objectList[0].ToString(), out double result))
            {
                DoublesCollection collection = new DoublesCollection() { Values = new List<double>() };
                foreach (string item in objectList.Select(x => x.ToString()))
                    collection.Values.Add(double.Parse(item));

                hourlyValues.Add(collection);
            }
            else
            {
                StringsCollection collection = new StringsCollection() {  Values = objectList.Select(x => x.ToString()).ToList() };
                hourlyValues.Add(collection);
            }
            return new oM.LadybugTools.HourlyContinuousCollection()
            {
                Values = hourlyValues[0],
                Header = ToHeader(oldObject["header"] as Dictionary<string, object>)
            };
        }

        public static string FromHourlyContinuousCollection(BH.oM.LadybugTools.HourlyContinuousCollection collection)
        {
            string valuesAsString = null;
            if (collection.Values.GetType() == typeof(DoublesCollection))
                valuesAsString = string.Join(", ", (collection.Values as DoublesCollection).Values.Select(x => x.ToString()));
            else if (collection.Values.GetType() == typeof(StringsCollection))
                valuesAsString = "\"" + string.Join("\", \"", (collection.Values as StringsCollection).Values) + "\"";

            string type = @"""type"" : ""HourlyContinuous""";
            string values = @"""values"" : [ " + valuesAsString + " ]";
            string header = @"""header"" : " + FromHeader(collection.Header).ToJson();

            return "{ " + type + ", " + values + ", " + header + " }";
        }
    }
}
