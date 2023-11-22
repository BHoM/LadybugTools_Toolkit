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
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.DataType ToDataType(Dictionary<string, object> oldObject)
        {
            string baseUnit;
            if (oldObject.ContainsKey("base_unit"))
                baseUnit = (string)oldObject["base_unit"];
            else
                baseUnit = "";
            try
            {
                return new oM.LadybugTools.DataType()
                {
                    Data_Type = (string)oldObject["data_type"],
                    Name = (string)oldObject["name"],
                    BaseUnit = baseUnit
                };
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error ocurred during conversion of a {typeof(DataType).FullName}. Returning a default {typeof(DataType).FullName}:\n The error: {ex}");
                return new DataType();
            }
        }

        public static Dictionary<string, object> FromDataType(BH.oM.LadybugTools.DataType dataType)
        {
            Dictionary<string, object> returnDict = new Dictionary<string, object>
            {
                { "type", "GenericDataType" },
                { "name", dataType.Name },
                { "data_type", dataType.Data_Type }
            };

            if (dataType.BaseUnit != "")
                returnDict.Add("base_unit", dataType.BaseUnit);

            return returnDict;
        }
    }
}
