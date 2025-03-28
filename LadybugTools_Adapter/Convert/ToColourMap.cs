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

using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ColourMap ToColourMap(this string colourMap)
        {
            if (colourMap == null)
            {
                BH.Engine.Base.Compute.RecordError("Cannot convert null string to a colourmap");
                return ColourMap.Undefined;
            }

            foreach (ColourMap item in Enum.GetValues(typeof(ColourMap)))
            {
                if (item.FromColourMap().ToLower() == colourMap.ToLower())
                    return item;

                if (item.ToString().ToLower() == colourMap.ToLower())
                    return item;

                FieldInfo field = item.GetType().GetField(item.ToString());
                DisplayTextAttribute[] array = field.GetCustomAttributes(typeof(DisplayTextAttribute), inherit: false) as DisplayTextAttribute[];
                if (array != null && array.Length > 0)
                    if (array.First().Text.ToLower() == colourMap.ToLower())
                        return item;
            }
            BH.Engine.Base.Compute.RecordError($"Could not convert the input string: {colourMap} to a colourmap.");
            return ColourMap.Undefined;
        }
    }
}

