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

using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

using BH.Engine.Environment;
using BH.Engine.Geometry;
using BH.oM.Base.Attributes;
using BH.oM.Environment.Elements;
using BH.oM.Geometry;
using BH.oM.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts an Environment Panel to a LadybugTools Shelter.")]
        [Input("panel", "Environment Panel to be converted into a shelter.")]
        [Input("radiationPorosity", "Radiation porosity for this shelter. (0-1).")]
        [Input("windPorosity", "Wind porosity for this shelter. (0-1).")]
        [Output("shelter", "LadybugTools Shelter object.")]
        public static Shelter ToShelter(this Panel panel, List<double> radiationPorosity = null, List<double> windPorosity = null)
        {
            if (panel == null)
            {
                Base.Compute.RecordError($"{nameof(panel)} is null. Shelter cannot be created.");
                return null;
            }

            if (radiationPorosity == null)
            {
                radiationPorosity = Enumerable.Repeat(0.0, 8760).ToList();
            }

            if (windPorosity == null)
            {
                windPorosity = Enumerable.Repeat(0.0, 8760).ToList();
            }

            List<Point> vertices = panel.Vertices().ToList();
            vertices.RemoveAt(vertices.Count - 1); // python Shelter object doesn't want a closed polyline

            return BH.Engine.LadybugTools.Create.Shelter(vertices: vertices, windPorosity: windPorosity, radiationPorosity: radiationPorosity);
        }
    }
}


