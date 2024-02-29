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
        public static string ToValidString(this ColourMap colourMap)
        {
            switch (colourMap)
            {
                case ColourMap.Undefined:
                    return "";
                case ColourMap.Viridis:
                    return "viridis";
                case ColourMap.Plasma:
                    return "plasma";
                case ColourMap.Inferno:
                    return "inferno";
                case ColourMap.Magma:
                    return "magma";
                case ColourMap.Cividis:
                    return "cividis";
                case ColourMap.YellowOrangeBrown:
                    return "YlOrBr";
                case ColourMap.YellowOrangeRed:
                    return "YlOrRd";
                case ColourMap.OrangeRed:
                    return "OrRd";
                case ColourMap.PurpleRed:
                    return "PuRd";
                case ColourMap.RedPurple:
                    return "RdPu";
                case ColourMap.BluePurple:
                    return "BuPu";
                case ColourMap.GreenBlue:
                    return "GnBu";
                case ColourMap.PurpleBlue:
                    return "PuBu";
                case ColourMap.YellowGreenBlue:
                    return "YlGnBu";
                case ColourMap.PurpleBlueGreen:
                    return "PuBuGn";
                case ColourMap.BlueGreen:
                    return "BuGn";
                case ColourMap.YellowGreen:
                    return "YlGn";
                case ColourMap.Binary:
                    return "binary";
                case ColourMap.GistYarg:
                    return "gist_yarg";
                case ColourMap.Grey:
                    return "gray";
                case ColourMap.Bone:
                    return "bone";
                case ColourMap.Pink:
                    return "pink";
                case ColourMap.Spring:
                    return "spring";
                case ColourMap.Summer:
                    return "summer";
                case ColourMap.Autumn:
                    return "autumn";
                case ColourMap.Winter:
                    return "winter";
                case ColourMap.Cool:
                    return "cool";
                case ColourMap.Hot:
                    return "hot";
                case ColourMap.AFMHot:
                    return "afmhot";
                case ColourMap.GistHeat:
                    return "gist_heat";
                case ColourMap.Copper:
                    return "copper";
                case ColourMap.PinkYellowGreen:
                    return "PiYG";
                case ColourMap.PinkRedGreen:
                    return "PRGn";
                case ColourMap.BrownBlueGreen:
                    return "BrBG";
                case ColourMap.PurpleOrange:
                    return "PuOr";
                case ColourMap.RedGrey:
                    return "RdGy";
                case ColourMap.RedBlue:
                    return "RdBu";
                case ColourMap.RedYellowBlue:
                    return "RdYlBu";
                case ColourMap.RedYellowGreen:
                    return "RdYlGn";
                case ColourMap.CoolWarm:
                    return "coolwarm";
                case ColourMap.BWR:
                    return "bwr";
                case ColourMap.Seismic:
                    return "seismic";
                case ColourMap.Twilight:
                    return "twilight";
                case ColourMap.TwilightShifted:
                    return "twilight_shifted";
                case ColourMap.HSV:
                    return "hsv";
                case ColourMap.Pastel:
                    return "Pastel1";
                case ColourMap.AlternatePastel:
                    return "Pastel2";
                case ColourMap.Dark:
                    return "Dark2";
                case ColourMap.Set1:
                    return "Set1";
                case ColourMap.Set2:
                    return "Set2";
                case ColourMap.Set3:
                    return "Set3";
                case ColourMap.Tab10:
                    return "tab10";
                case ColourMap.Tab20:
                    return "tab20";
                case ColourMap.Tab20b:
                    return "tab20b";
                case ColourMap.Tab20c:
                    return "tab20c";
                case ColourMap.Flag:
                    return "flag";
                case ColourMap.Prism:
                    return "prism";
                case ColourMap.Ocean:
                    return "ocean";
                case ColourMap.GistEarth:
                    return "gist_earth";
                case ColourMap.Terrain:
                    return "terrain";
                case ColourMap.GistStern:
                    return "gist_stern";
                case ColourMap.GNUPlot:
                    return "gnuplot";
                case ColourMap.AlternateGNUPlot:
                    return "gnuplot2";
                case ColourMap.CMRmap:
                    return "CMRmap";
                case ColourMap.CubeHelix:
                    return "cubehelix";
                case ColourMap.BlueRedGreen:
                    return "brg";
                case ColourMap.GistRainbow:
                    return "gist_rainbow";
                case ColourMap.Rainbow:
                    return "rainbow";
                case ColourMap.Jet:
                    return "jet";
                case ColourMap.Turbo:
                    return "turbo";
                case ColourMap.NipySpectral:
                    return "nipy_spectral";
                case ColourMap.GistNcar:
                    return "gist_ncar";
                case ColourMap.Greys:
                case ColourMap.Greens:
                case ColourMap.Blues:
                case ColourMap.Purples:
                case ColourMap.Oranges:
                case ColourMap.Reds:
                case ColourMap.Wistia:
                case ColourMap.Paired:
                case ColourMap.Spectral:
                case ColourMap.Accent:
                default:
                    return colourMap.ToString();

            }
        }

        public static ColourMap ToColourMap(this string colourMap)
        {
            if (colourMap == null)
            {
                BH.Engine.Base.Compute.RecordError("Cannot convert null string to a colourmap");
                return ColourMap.Undefined;
            }
            
            foreach (ColourMap item in Enum.GetValues(typeof(ColourMap)))
            {
                List<string> possibleValues = new List<string>();
                possibleValues.Add(item.ToString().ToLower());
                FieldInfo field = item.GetType().GetField(item.ToString());
                DisplayTextAttribute[] array = field.GetCustomAttributes(typeof(DisplayTextAttribute), inherit:false) as DisplayTextAttribute[];
                if (array != null && array.Length > 0)
                    possibleValues.Add(array.First().Text.ToLower());

                if (possibleValues.Any(x => x == colourMap.ToLower()))
                    return item;
            }
            BH.Engine.Base.Compute.RecordError($"Could not convert the input string: {colourMap} to a colourmap.");
            return ColourMap.Undefined;
        }
    }
}