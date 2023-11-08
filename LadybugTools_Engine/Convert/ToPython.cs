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

using BH.oM.Base.Attributes;
using System.ComponentModel;
using System.Threading;
using System.Linq;
using System.Collections.Generic;
using BH.oM.Geometry;
using BH.Engine.Serialiser;
using System.Web.Script.Serialization;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Convert this object to its Ladybug JSON string.")]
        [Input("point", "A Point object.")]
        [Output("str", "The JSON string representation of this BHoM object, ready for deserialistion in Python using Ladybug.")]
        public static string ToPythonString(this Point point)
        {
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            var dict = serializer.Deserialize<Dictionary<string, object>>(point.ToJson());
            dict["Type"] = "Point3D";
            var dictConverted = ToSnakeCase(dict);
            string json = serializer.Serialize(dictConverted);
            return json;
        }

        [Description("Convert this object to its Ladybug JSON string.")]
        [Input("collection", "A HourlyContinuousCollection object.")]
        [Output("str", "The JSON string representation of this BHoM object, ready for deserialistion in Python using Ladybug.")]
        public static string ToPythonString(this HourlyContinuousCollection collection)
        {
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            var dict = serializer.Deserialize<Dictionary<string, object>>(collection.ToJson());
            var dictConverted = ToSnakeCase(dict);
            string json = serializer.Serialize(dictConverted);
            return json.Replace("data__type", "data_type");
        }
    }
}
