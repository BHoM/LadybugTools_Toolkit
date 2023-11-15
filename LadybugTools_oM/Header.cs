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


using BH.oM.Base;
using System.Collections.Generic;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class Header : BHoMObject, ILadybugTools
    {
        [Description("The Ladybug datatype of this object, used for deserialisation.")]
        public virtual string Type { get; set; } = "Header";
        
        [Description("The data type the data associated with this header object represents.")]
        public virtual DataType DataType { get; set; } = new DataType();
        
        [Description("The unit for this header object.")]
        public virtual string Unit { get; set; } = string.Empty;
        
        [Description("The analysis period associated with this header object.")]
        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        [Description("The metadata associated with this header object.")]
        public virtual Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
    }
}
