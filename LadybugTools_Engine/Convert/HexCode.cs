/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
using System.ComponentModel;
using BH.oM.Base.Attributes;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts a colour to its respective RGB hexadecimal code (eg. white => #ffffff).")]
        [Input("colour", "The colour to convert into a hex code.")]
        [Output("hex", "The corresponding hex code.")]
        [PreviousVersion("7.2", "BH.Engine.LadybugTools.ToHexCode(System.Drawing.Color)")]
        public static string ToHexCode(this Color colour)
        {
            return $"#{colour.R.ToString("X2")}{colour.G.ToString("X2")}{colour.B.ToString("X2")}";
        }

        /**************************************************/

        [Description("Converts a string that is in the RGB hexadecimal format into a colour. (eg. #ffffff => white).")]
        [Input("hex", "The hexadecimal representation of a colour.")]
        [Output("colour", "The corresponding colour.")]
        [PreviousVersion("7.2", "BH.Engine.LadybugTools.FromHexCode(System.String)")]
        public static Color? FromHexCode(this string hex)
        {
            if (hex.IsNullOrEmpty())
            {
                BH.Engine.Base.Compute.RecordError("Cannot create a colour from an empty string.");
                return null;
            }

            //Convert to lower case, as both capital and lowercase a-f is also valid for hexadecimal.
            hex = hex.ToLower();

            //Allow # to be at the start of a hex string, as it is common and valid, but not necessary for getting the value.
            if (hex[0] == '#')
            {
                hex = hex.Substring(1);
            }

            if (hex.Length != 6)
            {
                BH.Engine.Base.Compute.RecordError($"The input string: {hex}, is invalid to create a colour from. Hex codes must be 6 characters long in Hexadecimal format.");
                return null;
            }

            List<char> chars = new List<char>() { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

            if (!hex.All(x => chars.Contains(x)))
            {
                BH.Engine.Base.Compute.RecordError($"The input string: {hex}, is invalid to create a colour from. Hex codes must be 6 characters long, and only contain numbers (0-9) and the characters a to f.");
                return null;
            }

            return Color.FromArgb(
                red: int.Parse(hex.Substring(0, 2), NumberStyles.HexNumber),
                green: int.Parse(hex.Substring(2, 2), NumberStyles.HexNumber),
                blue: int.Parse(hex.Substring(4, 2), NumberStyles.HexNumber));
        }
    }
}

