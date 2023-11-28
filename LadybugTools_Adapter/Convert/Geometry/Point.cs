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

using System;
using System.Collections.Generic;
using System.Text;
using BH.oM.LadybugTools;
using BH.oM.Geometry;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static Point ToPoint(Dictionary<string, object> oldObject)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;

            try
            {
                x = (double)oldObject["x"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading x of the Point. returning x as default ({x}).\n The error: {ex}");
            }

            try
            {
                y = (double)oldObject["y"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading y of the Point. returning y as default ({y}).\n The error: {ex}");
            }

            try
            {
                z = (double)oldObject["z"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading z of the Point. returning z as default ({z}).\n The error: {ex}");
            }

            return new Point()
            {
                X = x,
                Y = y,
                Z = z
            };
        }

        public static string FromPoint(Point point)
        {
            string type = @"""Point3D""";
            string xyz = $@"""x"" : {point.X}, ""y"" : {point.Y}, ""z"" : {point.Z}";
            return @"{""type"" : " + type + ", " + xyz + "}";
        }
    }
}
