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

using BH.oM.Base.Attributes;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [Description("Matplotlib standard colourmaps. This is not an exhaustive list of possible maps.")]
    public enum ColourMap
    {
        Undefined,
        Viridis,
        Plasma,
        Inferno,
        Magma,
        Cividis,
        Greys,
        Purples,
        Blues,
        Greens,
        Oranges,
        Reds,
        [DisplayText("Yellow, Orange, Brown")]
        YellowOrangeBrown,
        [DisplayText("Yellow, Orange, Red")]
        YellowOrangeRed,
        [DisplayText("Orange, Red")]
        OrangeRed,
        [DisplayText("Purple, Red")]
        PurpleRed,
        [DisplayText("Red, Purple")]
        RedPurple,
        [DisplayText("Blue, Purple")]
        BluePurple,
        [DisplayText("Green, Blue")]
        GreenBlue,
        [DisplayText("Purple, Blue")]
        PurpleBlue,
        [DisplayText("Yellow, Green, Blue")]
        YellowGreenBlue,
        [DisplayText("Purple, Blue, Green")]
        PurpleBlueGreen,
        [DisplayText("Blue, Green")]
        BlueGreen,
        [DisplayText("Yellow, Green")]
        YellowGreen,
        Binary,
        [DisplayText("Gist Yarg")]
        GistYarg,
        Grey,
        Bone,
        Pink,
        Spring,
        Summer,
        Autumn,
        Winter,
        Cool,
        Wistia,
        Hot,
        [DisplayText("AFM Hot")]
        AFMHot,
        [DisplayText("Gist Heat")]
        GistHeat,
        [DisplayText("Copper")]
        Copper,
        [DisplayText("Pink, Yellow, Green")]
        PinkYellowGreen,
        [DisplayText("Pink, Red, Green")]
        PinkRedGreen,
        [DisplayText("Brown, Blue, Green")]
        BrownBlueGreen,
        [DisplayText("Purple, Orange")]
        PurpleOrange,
        [DisplayText("Red, Grey")]
        RedGrey,
        [DisplayText("Red, Blue")]
        RedBlue,
        [DisplayText("Red, Yellow, Blue")]
        RedYellowBlue,
        [DisplayText("Red, Yellow, Green")]
        RedYellowGreen,
        Spectral,
        [DisplayText("Cool, warm")]
        CoolWarm,
        BWR,
        Seismic,
        Twilight,
        [DisplayText("Shifted Twilight")]
        TwilightShifted,
        HSV,
        Pastel,
        [DisplayText("Alternate Pastel")]
        AlternatePastel,
        Paired,
        Accent,
        Dark,
        Set1,
        Set2,
        Set3,
        [DisplayText("Tab 10")]
        Tab10,
        [DisplayText("Tab 20a")]
        Tab20,
        [DisplayText("Tab 20b")]
        Tab20b,
        [DisplayText("Tab 20c")]
        Tab20c,
        Flag,
        Prism,
        Ocean,
        [DisplayText("Gist Earth")]
        GistEarth,
        Terrain,
        [DisplayText("Gist Stern")]
        GistStern,
        [DisplayText("GNU Plot")]
        GNUPlot,
        [DisplayText("Alternate GNU Plot")]
        AlternateGNUPlot,
        [DisplayText("CMR Map")]
        CMRmap,
        [DisplayText("Cube Helix")]
        CubeHelix,
        [DisplayText("Blue, Red, Green")]
        BlueRedGreen,
        [DisplayText("Gist Rainbow")]
        GistRainbow,
        Rainbow,
        Jet,
        Turbo,
        [DisplayText("Nipy Spectral")]
        NipySpectral,
        [DisplayText("Gist Ncar")]
        GistNcar,
    };
}

