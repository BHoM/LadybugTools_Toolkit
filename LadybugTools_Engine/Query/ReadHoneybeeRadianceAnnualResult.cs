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
using System.IO;
using BH.oM.Adapters.LadybugTools;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Reads Radiance results from a directory")]
        [Input("directory", "A directory created using Queenbee, containing Radiance results.")]
        [Output("environmentObject", "Either an Environment Panel or an Environment Opening depending on the Honeybee Surface type.")]
        public static AnnualRadianceResult ReadHoneybeeRadianceAnnualResult(string directory)
        {
            AnnualRadianceResult result = new AnnualRadianceResult() { Directory = directory };

            // Get all ILL results files in directory
            IEnumerable<string> illFiles = Directory.EnumerateFiles(Path.Combine(directory, "results"), "*.ill", SearchOption.AllDirectories);

            // Get all points file in directory
            IEnumerable<string> ptsFiles = Directory.EnumerateFiles(Path.Combine(directory, "model"), "*.pts", SearchOption.AllDirectories);

            // Get sun-up hours in directory
            string sunUpHoursFile = Directory.EnumerateFiles(Path.Combine(directory, "results"), "sun-up-hours.txt", SearchOption.TopDirectoryOnly).First();

            // Load points and vectors into grid object-ish thing
            foreach (string ptsFile in ptsFiles)
            {
                List<Point> grid_pts = new List<Point>();
                List<Vector> grid_vecs = new List<Vector>();
                foreach (string ptString in File.ReadLines(ptsFile).ToList())
                {
                    string[] attrs = ptString.Split(null);
                    grid_pts.Add(new Point() { X = System.Convert.ToDouble(attrs[0]), Y = System.Convert.ToDouble(attrs[1]), Z = System.Convert.ToDouble(attrs[2]) });
                    grid_vecs.Add(new Vector() { X = System.Convert.ToDouble(attrs[3]), Y = System.Convert.ToDouble(attrs[4]), Z = System.Convert.ToDouble(attrs[5]) });
                }
                result.Points.Add(grid_pts);
                result.Vectors.Add(grid_vecs);
            }

            // Load hourly results into list

            // Up-sample list into hourly annual list

            // Return the combined object

            return result;
        }

    }
}

