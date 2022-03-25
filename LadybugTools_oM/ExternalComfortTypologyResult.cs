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

using System.Collections.Generic;
using System.ComponentModel;
using BH.oM.Base;

namespace BH.oM.Ladybug
{
    public class ExternalComfortTypologyResult : BHoMObject
    {
        [Description("The typology to which results are associated.")]
        public virtual ExternalComfortTypology ExternalComfortTypology { get; set; } = ExternalComfortTypology.Undefined;

        [Description("The results object containing the simulation results for this typology.")]
        public virtual ExternalComfortResult ExternalComfortResult { get; set; } = new ExternalComfortResult();

        [Description("The effective dry bulb temperature for this typology under the given ExternalComfortResult.")]
        public virtual List<double> DryBulbTemperature { get; set; } = new List<double>();

        [Description("The effective relative humidity for this typology under the given ExternalComfortResult.")]
        public virtual List<double> RelativeHumidity { get; set; } = new List<double>();

        [Description("The effective wind speed for this typology under the given ExternalComfortResult.")]
        public virtual List<double> WindSpeed { get; set; } = new List<double>();

        [Description("The effective mean radiant temperature for this typology under the given ExternalComfortResult.")]
        public virtual List<double> MeanRadiantTemperature { get; set; } = new List<double>();
    }
}
