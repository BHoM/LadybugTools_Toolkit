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

using BH.Engine.Geometry;
using BH.oM.Base.Attributes;
using BH.oM.Geometry;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Shelter object.")]
        [Input("vertices", "The vertices of the shelter.")]
        [Input("windPorosity", "The hourly wind porosity of the shelter (0-1).")]
        [Input("radiationPorosity", "The hourly radiation porosity of the shelter (0-1).")]
        [Output("shelter", "A Shelter object.")]
        public static Shelter Shelter(List<Point> vertices, List<double> windPorosity = null, List<double> radiationPorosity = null)
        {
            if ((windPorosity.Count() == 0 && windPorosity.Sum() == 0) || windPorosity == null)
                windPorosity = Enumerable.Repeat(0.0, 8760).ToList();

            if (windPorosity.Count != 8760)
            {
                BH.Engine.Base.Compute.RecordError("windPorosity must be a list of 8760 values");
                return null;
            }
            if (windPorosity.Where(x => x < 0 || x > 1).Any())
            {
                BH.Engine.Base.Compute.RecordError("Shelter wind porosity must be between 0 and 1.");
                return null;
            }

            if ((radiationPorosity.Count() == 0 && radiationPorosity.Sum() == 0) || radiationPorosity == null)
                radiationPorosity = Enumerable.Repeat(0.0, 8760).ToList();

            if (radiationPorosity.Count != 8760)
            {
                BH.Engine.Base.Compute.RecordError("radiationPorosity must be a list of 8760 values");
                return null;
            }

            if (radiationPorosity.Where(x => x < 0 || x > 1).Any())
            {
                BH.Engine.Base.Compute.RecordError("Shelter radiation porosity must be between 0 and 1.");
                return null;
            }

            if (radiationPorosity.Sum() + windPorosity.Sum() == 8760 * 2)
            {
                BH.Engine.Base.Compute.RecordError("This shelter is completely transmissive and would have no effect.");
                return null;
            }

            if (!BH.Engine.Geometry.Create.Polyline(vertices).IsPlanar())
            {
                BH.Engine.Base.Compute.RecordError("Shelter vertices are not planar.");
                return null;
            }

            return new Shelter()
            {
                Vertices = vertices,
                WindPorosity = windPorosity,
                RadiationPorosity = radiationPorosity
            };
        }
    }
}
