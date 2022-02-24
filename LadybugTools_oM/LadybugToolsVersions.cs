/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2022, the respective contributors. All rights reserved.
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


using BH.oM.Base;

using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class LadybugToolsVersions : BHoMObject
    {
        [Description("The most recent version of lbt-dragonfly.")]
        public virtual string LbtDragonfly { get; set; } = "0.8.367";

        [Description("The most recent version of lbt-recipes")]
        public virtual string LbtRecipes { get; set; } = "0.19.4";

        [Description("The most recent version of ladybug-rhino")]
        public virtual string LadybugRhino { get; set; } = "1.33.3";

        [Description("The most recent version of lbt-grasshopper")]
        public virtual string LbtGrasshopper { get; set; } = "1.4.0";

        [Description("The most recent version of ladybug-grasshopper-dotnet.")]
        public virtual string LadybugGrasshopperDotnet { get; set; } = "1.1.3";

        [Description("The most recent version of honeybee-openstudio-gem.")]
        public virtual string HoneybeeOpenstudioGem { get; set; } = "2.28.6";

        [Description("The most recent version of lbt-measures.")]
        public virtual string LbtMeasures { get; set; } = "0.2.0";

        [Description("The most recent version of honeybee-standards.")]
        public virtual string HoneybeeStandards { get; set; } = "2.0.5";

        [Description("The most recent version of honeybee-energy-standards.")]
        public virtual string HoneybeeEnergyStandards { get; set; } = "2.2.4";
    }
}
