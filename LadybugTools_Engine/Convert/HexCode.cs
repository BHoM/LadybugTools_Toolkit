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

using BH.Engine.Base;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Globalization;

namespace BH.Engine.LadyBugTools
{
    public static partial class Convert
    {
        public static string ToHexCode(this Color colour)
        {
            return $"#{colour.R.ToString("X2")}{colour.G.ToString("X2")}{colour.B.ToString("X2")}";
        }

        public static Color? FromHexCode(this string colour)
        {
            if (colour.IsNullOrEmpty())
            {
                BH.Engine.Base.Compute.RecordError("Cannot create a colour from an empty string.");
                return null;
            }

            colour = colour.ToLower();

            if (colour[0] == '#')
            {
                colour = colour.Substring(1);
            }

            if (colour.Length != 6)
            {
                BH.Engine.Base.Compute.RecordError($"The input string: {colour}, is invalid to create a colour from. Hex codes must be 6 characters long, and only contain numbers (0-9) and the characters a to f.");
                return null;
            }

            List<char> chars = new List<char>() { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

            if (!colour.All(x => chars.Contains(x)))
            {
                BH.Engine.Base.Compute.RecordError($"The input string: {colour}, is invalid to create a colour from. Hex codes must be 6 characters long, and only contain numbers (0-9) and the characters a to f.");
                return null;
            }

            return Color.FromArgb(
                red: int.Parse(colour.Substring(0, 2), NumberStyles.HexNumber),
                green: int.Parse(colour.Substring(2, 2), NumberStyles.HexNumber),
                blue: int.Parse(colour.Substring(4, 2), NumberStyles.HexNumber));
        }
    }
}
