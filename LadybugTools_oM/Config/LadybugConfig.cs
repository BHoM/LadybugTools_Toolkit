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

using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("The action config for the LadybugTools Adapter.")]
    public class LadybugConfig : ActionConfig
    {
        [Description("File settings for the json file to pull/push to.")]
        public virtual FileSettings JsonFile { get; set; } = null;

        [Description("The amount of time (in days) any files that have been created by the adapter for caching purposes should exist before being removed/recreated. \n Files are only deleted/updated . \n Set to 0 to force a recompute of a simulation that has a stored cache.")]
        public virtual int CacheFileMaximumAge { get; set; } = 30;
    }
}