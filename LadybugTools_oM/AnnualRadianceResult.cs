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

using BH.oM.Base;
using BH.oM.Geometry;
using BH.oM.Reflection;
using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.Adapters.LadybugTools
{
    public class AnnualRadianceResult : BHoMObject
    {
        [Description("Directory where the Annual Radiance Result is stored.")]
        public virtual string Directory { get; set; } = "";

        [Description("Sensor locations - one list of points per grid.")]
        public virtual List<List<Point>> Points { get; set; } = new List<List<Point>>();
        
        [Description("Sensor vectors - one list of vectors per grid.")]
        public virtual List<List<Vector>> Vectors { get; set; } = new List<List<Vector>>();

        [Description("Results values - one per grid, per time-step, per Point.")]
        public virtual List<List<List<double>>> Results { get; set; } = new List<List<List<double>>>();
    }
}

