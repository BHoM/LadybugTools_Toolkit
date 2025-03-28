/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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
            Header header = new Header();
            List<double?> hourlyValues = new List<double?>();
            try
            {
                if (oldObject["header"].GetType() == typeof(CustomObject))
                    oldObject["header"] = (oldObject["header"] as CustomObject).CustomData;
                header = ToHeader(oldObject["header"] as Dictionary<string, object>);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the Header of the HourlyContinuousCollection. returning a default Header.\n The error: {ex}");
            }

            try
            {
                hourlyValues = (oldObject["values"] as List<object>).Select(x => x == null ? null : System.Convert.ToDouble(x) as double?).ToList();
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when converting the values in a collection. Returning an empty collection: \n The error: {ex}");
            }

            return new oM.LadybugTools.HourlyContinuousCollection()
            {
                Values = hourlyValues,
                Header = header
            };
        }

        public static string FromHourlyContinuousCollection(BH.oM.LadybugTools.HourlyContinuousCollection collection)
        {
            string valuesAsString = string.Join(", ", collection.Values);

            string type = @"""type"" : ""HourlyContinuous""";
            string values = $@"""values"" : [{valuesAsString}]";
            string header = $@"""header"" : {FromHeader(collection.Header).ToJson()}";

            return "{ " + type + ", " + values + ", " + header + " }";
        }
    }
}


