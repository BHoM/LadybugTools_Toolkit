/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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
using System.Text;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using BH.oM.Geometry;
using BH.Engine.Geometry;
using BH.oM.Dimensional;
using BH.oM.Analytical.Elements;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a Shelter from geometry.")]
        [Input("polyline", "A planar Polyline.")]
        [Input("windPorosity", "The porosity of this shelter object to wind.")]
        [Input("radiationPorosity", "The porosity of this shelter object to radiation.")]
        [Output("shelter", "A shelter object.")]
        public static Shelter Shelter(Polyline polyline, double windPorosity = 0, double radiationPorosity = 0)
        {
            if (!polyline.IsPlanar())
            {
                BH.Engine.Base.Compute.RecordError("Input object is not planar.");
            }

            if (!polyline.IsClosed())
            {
                BH.Engine.Base.Compute.RecordError("Input object is not closed.");
            }

            List<List<double>> vertices = new List<List<double>>();
            for (int i = 0; i < polyline.ControlPoints.Count - 1; i++)
            {
                vertices.Add(new List<double>() { polyline.ControlPoints[i].X, polyline.ControlPoints[i].Y, polyline.ControlPoints[i].Z });
            }

            return new Shelter() { Vertices = vertices, WindPorosity = windPorosity, RadiationPorosity = radiationPorosity };
        }

        [Description("Create a Shelter from geometry.")]
        [Input("surface", "A planar surface.")]
        [Input("windPorosity", "The porosity of this shelter object to wind.")]
        [Input("radiationPorosity", "The porosity of this shelter object to radiation.")]
        [Output("shelter", "A shelter object.")]
        public static Shelter Shelter(PlanarSurface surface, double windPorosity = 0, double radiationPorosity = 0)
        {
            return Shelter(BH.Engine.Geometry.Create.Polyline(surface.ExternalBoundary.IControlPoints()), windPorosity, radiationPorosity);
        }
    }
}
