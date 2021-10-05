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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BH.oM.Geometry;
using BH.oM.Environment;
using BH.oM.Environment.Elements;

using BH.Engine.Geometry;
using BH.Engine.Environment;

using BH.oM.Reflection.Attributes;
using System.ComponentModel;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts a Ladybug style object to a JSON serialised version of that object.")]
        [Input("header", "LadybugTools Header object.")]
        [Output("json", "A JSON string.")]
        public static string ToJSON(this Header header)
        {
            // TODO - finish method
            return "";
        }

        [Description("Converts a Ladybug style object to a JSON serialised version of that object.")]
        [Input("analysisPeriod", "LadybugTools AnalysisPeriod object.")]
        [Output("json", "A JSON string.")]
        public static string ToJSON(this AnalysisPeriod analysisPeriod)
        {
            // TODO - finish method
            return "";
        }

        

        [Description("Converts a Ladybug style object to a JSON serialised version of that object.")]
        [Input("hourlyContinuousCollection", "LadybugTools HourlyContinuousCollection object.")]
        [Output("json", "A JSON string.")]
        public static string ToJSON(this HourlyContinuousCollection hourlyContinuousCollection)
        {
            // TODO - finish method
            return "";
        }
    }
}

