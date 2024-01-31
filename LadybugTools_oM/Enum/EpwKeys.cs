/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                     
 *                                      
 * The BHoM is free software: you can redistribute it and/or modify     
 * it under the terms of the GNU Lesser General Public License as published by_
 * the Free Software Foundation, either version 3.0 of the License, or    
 * (at your option) any later version.                    
 *                                      
 * The BHoM is distributed in the hope that it will be useful,      
 * but WITHOUT ANY WARRANTY; without even the implied warranty of       
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the         
 * GNU Lesser General Public License for more details.            
 *                                      
 * You should have received a copy of the GNU Lesser General Public License   
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.  _
 */

using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    [Description("An enum for keys that frequently appear in epw files.")]
    public enum EpwKey
    {
        Undefined,
        Aerosol_Optical_Depth,
        Atmospheric_Station_Pressure,
        Ceiling_Height,
        Days_Since_Last_Snowfall,
        Dew_Point_Temperature,
        Diffuse_Horizontal_Illuminance,
        Diffuse_Horizontal_Radiation,
        Direct_Normal_Illuminance,
        Direct_Normal_Radiation,
        Dry_Bulb_Temperature,
        Extraterrestrial_Direct_Normal_Radiation,
        Extraterrestrial_Horizontal_Radiation,
        Global_Horizontal_Illuminance,
        Global_Horizontal_Radiation,
        Horizontal_Infrared_Radiation_Intensity,
        Opaque_Sky_Cover,
        Precipitable_Water,
        Present_Weather_Codes,
        Present_Weather_Observation,
        Relative_Humidity,
        Snow_Depth,
        Total_Sky_Cover,
        Visibility,
        Wind_Direction,
        Wind_Speed,
        Zenith_Luminance
  };
}

