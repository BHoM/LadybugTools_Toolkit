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

using BH.oM.LadybugTools;
using BH.oM.Geometry;
using BH.oM.Reflection.Attributes;

using System.ComponentModel;
using System.IO;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Create a Honeybee-Radiance SensorGrid object from a file containing Positions and Vectors.")]
        [Input("ptsFile", "A files containing Points and Vectors, used in a Radiance simulation.")]
        [Output("sensorGrid", "A BHoM-HBRadiance SensorGrid.")]
        public static SensorGrid SensorGrid(string ptsFile)
        {
            SensorGrid sensorGrid = new SensorGrid
            {
                Name = Path.GetFileNameWithoutExtension(ptsFile)
            };

            foreach (string ptString in File.ReadLines(ptsFile).ToList())
            {
                string[] attrs = ptString.Split(null);
                sensorGrid.Positions.Add(
                    new Point() { 
                        X = System.Convert.ToDouble(attrs[0]), 
                        Y = System.Convert.ToDouble(attrs[1]), 
                        Z = System.Convert.ToDouble(attrs[2])
                    });
                sensorGrid.Directions.Add(
                    new Vector() { 
                        X = System.Convert.ToDouble(attrs[3]), 
                        Y = System.Convert.ToDouble(attrs[4]), 
                        Z = System.Convert.ToDouble(attrs[5])
                    });
            }

            return sensorGrid;
        }
    }
}
