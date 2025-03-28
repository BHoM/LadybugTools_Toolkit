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
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [Description("Matplotlib standard colourmaps. This is not an exhaustive list of possible maps.")]
    public enum ColourMap
    {
        Undefined,
        Accent,
        [DisplayText("AFM Hot")]
        AFMHot,
        [DisplayText("Alternate Pastel")]
        AlternatePastel,
        [DisplayText("Alternate GNU Plot")]
        AlternateGNUPlot,
        Autumn,
        Binary,
        [DisplayText("Blue, Green")]
        BlueGreen,
        [DisplayText("Blue, Purple")]
        BluePurple,
        [DisplayText("Blue, Red, Green")]
        BlueRedGreen,
        Blues,
        Bone,
        [DisplayText("Brown, Blue, Green")]
        BrownBlueGreen,
        BWR,
        Cividis,
        [DisplayText("CMR Map")]
        CMRmap,
        Cool,
        [DisplayText("Cool, warm")]
        CoolWarm,
        Copper,
        [DisplayText("Cube Helix")]
        CubeHelix,
        Dark,
        Flag,
        [DisplayText("Gist Rainbow")]
        GistRainbow,
        [DisplayText("Gist Yarg")]
        GistYarg,
        [DisplayText("Gist Heat")]
        GistHeat,
        [DisplayText("Gist Stern")]
        GistStern,
        [DisplayText("Gist Earth")]
        GistEarth,
        [DisplayText("Gist Ncar")]
        GistNcar,
        [DisplayText("GNU Plot")]
        GNUPlot,
        [DisplayText("Green, Blue")]
        GreenBlue,
        Greens,
        Grey,
        Greys,
        Hot,
        HSV,
        Inferno,
        Jet,
        Magma,
        [DisplayText("Nipy Spectral")]
        NipySpectral,
        Ocean,
        [DisplayText("Orange, Red")]
        OrangeRed,
        Oranges,
        Paired,
        Pastel,
        Pink,
        [DisplayText("Pink, Red, Green")]
        PinkRedGreen,
        [DisplayText("Pink, Yellow, Green")]
        PinkYellowGreen,
        Plasma,
        Prism,
        [DisplayText("Purple, Blue")]
        PurpleBlue,
        [DisplayText("Purple, Blue, Green")]
        PurpleBlueGreen,
        [DisplayText("Purple, Orange")]
        PurpleOrange,
        [DisplayText("Purple, Red")]
        PurpleRed,
        Purples,
        Rainbow,
        [DisplayText("Red, Blue")]
        RedBlue,
        [DisplayText("Red, Grey")]
        RedGrey,
        [DisplayText("Red, Purple")]
        RedPurple,
        Reds,
        [DisplayText("Red, Yellow, Blue")]
        RedYellowBlue,
        [DisplayText("Red, Yellow, Green")]
        RedYellowGreen,
        Seismic,
        Set1,
        Set2,
        Set3,
        Spring,
        Spectral,
        Summer,
        [DisplayText("Tab 10")]
        Tab10,
        [DisplayText("Tab 20a")]
        Tab20,
        [DisplayText("Tab 20b")]
        Tab20b,
        [DisplayText("Tab 20c")]
        Tab20c,
        Terrain,
        Turbo,
        Twilight,
        [DisplayText("Shifted Twilight")]
        TwilightShifted,
        Viridis,
        Winter,
        Wistia,
        [DisplayText("Yellow, Green")]
        YellowGreen,
        [DisplayText("Yellow, Green, Blue")]
        YellowGreenBlue,
        [DisplayText("Yellow, Orange, Brown")]
        YellowOrangeBrown,
        [DisplayText("Yellow, Orange, Red")]
        YellowOrangeRed,
    }
}


