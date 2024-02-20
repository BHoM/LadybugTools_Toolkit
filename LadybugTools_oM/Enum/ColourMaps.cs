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
        [DisplayText("Viridis")]
        viridis,
        [DisplayText("Plasma")]
        plasma,
        [DisplayText("Inferno")]
        inferno,
        [DisplayText("Magma")]
        magma,
        [DisplayText("Cividis")]
        cividis,
        Greys,
        Purples,
        Blues,
        Greens,
        Oranges,
        Reds,
        [DisplayText("Yellow, Orange, Brown")]
        YlOrBr,
        [DisplayText("Yellow, Orange, Red")]
        YlOrRd,
        [DisplayText("Orange, Red")]
        OrRd,
        [DisplayText("Purple, Red")]
        PuRd,
        [DisplayText("Red, Purple")]
        RdPu,
        [DisplayText("Blue, Purple")]
        BuPu,
        [DisplayText("Green, Blue")]
        GnBu,
        [DisplayText("Purplse, Blue")]
        PuBu,
        [DisplayText("Yellow, Green, Blue")]
        YlGnBu,
        [DisplayText("Purple, Blue, Green")]
        PuBuGn,
        [DisplayText("Blue, Green")]
        BuGn,
        [DisplayText("Yellow, Green")]
        YlGn,
        [DisplayText("Binary")]
        binary,
        [DisplayText("Gist Yarg")]
        gist_yarg,
        [DisplayText("Grey")]
        gray,
        [DisplayText("Bone")]
        bone,
        [DisplayText("Pink")]
        pink,
        [DisplayText("Spring")]
        spring,
        [DisplayText("Summer")]
        summer,
        [DisplayText("Autumn")]
        autumn,
        [DisplayText("Winter")]
        winter,
        [DisplayText("Cool")]
        cool,
        Wistia,
        [DisplayText("Hot")]
        hot,
        [DisplayText("AFM Hot")]
        afmhot,
        [DisplayText("Gist Heat")]
        gist_heat,
        [DisplayText("Copper")]
        copper,
        [DisplayText("Pink, Yellow, Green")]
        PiYG,
        [DisplayText("Pink, Red, Green")]
        PRGn,
        [DisplayText("Brown, Blue, Green")]
        BrBG,
        [DisplayText("Purple, Orange")]
        PuOr,
        [DisplayText("Red, Grey")]
        RdGy,
        [DisplayText("Red, Blue")]
        RdBu,
        [DisplayText("Red, Yellow, Blue")]
        RdYlBu,
        [DisplayText("Red, Yellow, Green")]
        RdYlGn,
        Spectral,
        [DisplayText("Cool, warm")]
        coolwarm,
        bwr,
        [DisplayText("Seismic")]
        seismic,
        [DisplayText("Twilight")]
        twilight,
        [DisplayText("Shifted Twilight")]
        twilight_shifted,
        hsv,
        [DisplayText("Pastel")]
        Pastel1,
        [DisplayText("Alternate Pastel")]
        Pastel2,
        Paired,
        Accent,
        [DisplayText("Dark")]
        Dark2,
        Set1,
        Set2,
        Set3,
        [DisplayText("Tab 10")]
        tab10,
        [DisplayText("Tab 20a")]
        tab20,
        [DisplayText("Tab 20b")]
        tab20b,
        [DisplayText("Tab 20c")]
        tab20c,
        [DisplayText("Flag")]
        flag,
        [DisplayText("Prism")]
        prism,
        [DisplayText("Ocean")]
        ocean,
        [DisplayText("Gist Earth")]
        gist_earth,
        [DisplayText("Terrain")]
        terrain,
        [DisplayText("Gist Stern")]
        gist_stern,
        [DisplayText("GNU Plot")]
        gnuplot,
        [DisplayText("Alternate GNU Plot")]
        gnuplot2,
        [DisplayText("CMR Map")]
        CMRmap,
        [DisplayText("Cube Helix")]
        cubehelix,
        [DisplayText("Blue, Red, Green")]
        brg,
        [DisplayText("Gist Rainbow")]
        gist_rainbow,
        [DisplayText("Rainbow")]
        rainbow,
        [DisplayText("Jet")]
        jet,
        [DisplayText("Turbo")]
        turbo,
        [DisplayText("Nipy Spectral")]
        nipy_spectral,
        [DisplayText("Gist Ncar")]
        gist_ncar,
    };
}

