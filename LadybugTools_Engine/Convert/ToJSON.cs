/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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

using System.ComponentModel;
using BH.oM.LadybugTools;
using BH.oM.Reflection.Attributes;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts an ILadybugObject to it's JSON string representation in order to exchange data with Ladybug Python.")]
        [Input("ladybugObject", "A BHoM Ladybug object.")]
        [Output("jsonString", "The JSON string representation of the BHoM Ladybug object.")]
        public static string ToJson(this ILadybugObject ladybugObject)
        {
            string json = JsonConvert.SerializeObject(ladybugObject, Formatting.Indented);
            JObject rss = JObject.Parse(json);
            // TODO - ADD LOGIC HERE TO HANDLE dATA_tYPE AND tYPE IN idATAtYPE DIFFERENTLY!
            rss.Add("type", ladybugObject.GetType().Name);
            return rss.ToString();
        }
    }
}

