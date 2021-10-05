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

using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class SensorGrid : BHoMObject
    {
        [Description("Sensor locations.")]
        public virtual List<Point> Positions { get; set; } = new List<Point>();
        
        [Description("Sensor vectors.")]
        public virtual List<Vector> Directions { get; set; } = new List<Vector>();

        [Description("An optional mesh that aligns with the sensors.")]
        public virtual Mesh3D Mesh { get; set; } = null;

        [Description("An optional brep for the geometry used to make the sensor grid.")]
        public virtual IGeometry BaseGeo { get; set; } = null;
    }
}
