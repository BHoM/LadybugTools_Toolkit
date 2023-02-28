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
using BH.oM.Environment.Elements;
using BH.oM.LadybugTools;
using System.Collections.Generic;
using System.ComponentModel;
using System;
using System.Threading;
using System.Linq;
using BH.Engine.Base;
using BH.Engine.Environment;
using System.Runtime.CompilerServices;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts an Environment Panel to a LadybugTools Shelter.")]
        [Input("panel", "Environment Panel to be converted into a shelter.")]
        [Input("radiationPorosity", "Radiation porosity for this shelter. (0-1).")]
        [Input("windPorosity", "Wind porosity for this shelter. (0-1).")]
        [Output("shelter", "LadybugTools Shelter object.")]
        public static Shelter ToShelter(this Panel panel, double radiationPorosity = 0, double windPorosity = 0)
        {
            if (panel == null)
            {
                Base.Compute.RecordError("Panel is null. Shelter cannot be created.");
                return null;
            }

            double[] vars = new[] { radiationPorosity, windPorosity };

            List<Tuple<string, double>> varList = new List<Tuple<string, double>>()
            {
                new Tuple<string, double>(nameof(radiationPorosity), radiationPorosity),
                new Tuple<string, double>(nameof(windPorosity), windPorosity)
            };

            foreach (var val in varList)
            {
                if (!val.Item2.IsBetween(0,1))
                {
                    Base.Compute.RecordError($"{val.Item1} must be between 0 and 1, but is actually {val.Item2}.");
                    return null;
                }
            }

            List<List<double>> vertices = panel.Vertices().Select(v => new List<double>() { v.X, v.Y, v.Z }).ToList();
            vertices.RemoveAt(vertices.Count - 1); // python Shelter object doesn't want a closed polyline

            return new Shelter()
            {
                Vertices = vertices,
                WindPorosity = windPorosity,
                RadiationPorosity = radiationPorosity
            };
        }

        private static bool IsBetween(this double val, double min, double max)
        {
            return (val >= min && val <= max);
        }
    }
}

