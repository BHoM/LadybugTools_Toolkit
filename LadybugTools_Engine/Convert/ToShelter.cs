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
                Base.Compute.RecordError("Panel is null. Shelter cannot be created.");
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

            if (radiationPorosity.Count != 8760)
            {
                Base.Compute.RecordError("Radiation porosity list must be 8760 long.");
                return null;
            }
            if (windPorosity.Count != 8760)
            {
                Base.Compute.RecordError("Wind porosity list must be 8760 long.");
                return null;
            }

            foreach (double value in radiationPorosity)
            {
                if (value <= 0 || value >= 1)
                {
                    Base.Compute.RecordError("Radiation porosity must be between 0-1 (inclusive).");
                    return null;
                }
            }
            foreach (double value in windPorosity)
            {
                if (value <= 0 || value >= 1)
                {
                    Base.Compute.RecordError("Wind porosity must be between 0-1 (inclusive).");
                    return null;
                }
            }

            if (windPorosity.Sum() + radiationPorosity.Sum() == 0)
            {
                Base.Compute.RecordError("This Shelter will have no effect as it is completely transmissive.");
                return null;
            }

            List<Point> vertices = panel.Vertices().ToList();
            vertices.RemoveAt(vertices.Count - 1); // python Shelter object doesn't want a closed polyline

            return new Shelter()
            {
                Vertices = vertices,
                WindPorosity = windPorosity,
                RadiationPorosity = radiationPorosity
            };
        }
    }
}
