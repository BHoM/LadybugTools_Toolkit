using BH.Engine.Serialiser;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.EPW ToEPW(Dictionary<string, object> oldObject)
        {
            if (oldObject["location"].GetType() == typeof(CustomObject))
            {
                oldObject["location"] = (oldObject["location"] as CustomObject).CustomData;
            }
            if (oldObject["metadata"].GetType() == typeof(CustomObject))
            {
                oldObject["metadata"] = (oldObject["metadata"] as CustomObject).CustomData;
            }
            List<BH.oM.LadybugTools.HourlyContinuousCollection> collections = new List<BH.oM.LadybugTools.HourlyContinuousCollection>();
            foreach (var collection in oldObject["data_collections"] as List<object>)
            {
                if (collection.GetType() == typeof(CustomObject))
                {
                    collections.Add(ToHourlyContinuousCollection((collection as CustomObject).CustomData));
                }
                else
                {
                    collections.Add(ToHourlyContinuousCollection(collection as Dictionary<string, object>));
                }
            }
            return new oM.LadybugTools.EPW()
            {
                Location = ToLocation(oldObject["location"] as Dictionary<string, object>),
                DataCollections = collections,
                Metadata = (Dictionary<string, object>)oldObject["metadata"]
            };
        }
        
        public static string FromEPW(BH.oM.LadybugTools.EPW epw)
        {
            string type = "EPW";
            string location = FromLocation(epw.Location).ToJson();
            string dataCollections = string.Join(", ", epw.DataCollections.Select(x => FromHourlyContinuousCollection(x)));
            string metadata = epw.Metadata.ToJson();
            if (metadata.Length == 0)
            {
                metadata = "{}";
            }
            string json = @"{ ""type"": """ + type + @""", ""location"": " + location + @", ""data_collections"": [ " + dataCollections + @"], ""metadata"": " + metadata + "}";
            return json;
        }
    }
}
