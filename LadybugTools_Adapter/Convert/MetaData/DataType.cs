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
            string dataType = "GenericType";
            string name = "";

            //base_unit only occurs when Data_Type is "GenericType".
            if (oldObject.ContainsKey("base_unit"))
                baseUnit = (string)oldObject["base_unit"];
            else
                baseUnit = "";

            try
            {
                dataType = (string)oldObject["data_type"];
            }
            catch (Exception ex)
            {
                //This DataType object won't work properly in ladybug python when serialised again as it required a base_unit.
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the data type of the DataType. returning data type as default ({dataType}).\n The error: {ex}");
            }

            try
            {
                name = (string)oldObject["name"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the name of the DataType. returning name as default (\"\").\n The error: {ex}");
            }

            return new oM.LadybugTools.DataType()
            {
                Data_Type = dataType,
                Name = name,
                BaseUnit = baseUnit
            };
        }

        public static Dictionary<string, object> FromDataType(BH.oM.LadybugTools.DataType dataType)
        {
            Dictionary<string, object> returnDict = new Dictionary<string, object>
            {
                { "name", dataType.Name },
                { "data_type", dataType.Data_Type }
            };
            
            string type;

            if (dataType.Data_Type == "GenericType")
            {
                type = "GenericDataType";
                returnDict.Add("bast_unit", dataType.BaseUnit);
            }
            else
            {
                type = "DataType";
            }
            returnDict.Add("type", type);

            return returnDict;
        }
    }
}
