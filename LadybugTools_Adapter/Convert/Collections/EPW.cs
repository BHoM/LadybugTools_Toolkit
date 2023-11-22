/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

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
                oldObject["location"] = (oldObject["location"] as CustomObject).CustomData;

            if (oldObject["metadata"].GetType() == typeof(CustomObject))
                oldObject["metadata"] = (oldObject["metadata"] as CustomObject).CustomData;

            List<BH.oM.LadybugTools.HourlyContinuousCollection> collections = new List<BH.oM.LadybugTools.HourlyContinuousCollection>();
            foreach (var collection in oldObject["data_collections"] as List<object>)
            {
                if (collection.GetType() == typeof(CustomObject))
                    collections.Add(ToHourlyContinuousCollection((collection as CustomObject).CustomData));
                else
                    collections.Add(ToHourlyContinuousCollection(collection as Dictionary<string, object>));
            }
            EPW epw = new EPW()
            {
                Location = ToLocation(oldObject["location"] as Dictionary<string, object>),
                DataCollections = collections
            };
            try
            {
                epw.Metadata = (Dictionary<string, object>)oldObject["metadata"];
                return epw;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred during conversion of Metadata, returning without Metadata:\n The error: {ex}");
                return epw;
            }
        }
        
        public static string FromEPW(BH.oM.LadybugTools.EPW epw)
        {
            string type = "EPW";
            string location = FromLocation(epw.Location).ToJson();
            string dataCollections = string.Join(", ", epw.DataCollections.Select(x => FromHourlyContinuousCollection(x)));
            string metadata = epw.Metadata.ToJson();

            if (metadata.Length == 0)
                metadata = "{}";

            string json = @"{ ""type"": """ + type + @""", ""location"": " + location + @", ""data_collections"": [ " + dataCollections + @"], ""metadata"": " + metadata + "}";

            return json;
        }
    }
}
