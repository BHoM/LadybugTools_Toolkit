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

using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.IDataType ToDataType(Dictionary<string, object> oldObject)
        {
            string name = "";
            string dataType = "";
            string unit = "";

            try
            {
                name = (string)oldObject["name"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the name of the DataType. returning name as default (\"\").\n The error: {ex}");
            }

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
                unit = (string)oldObject["unit"];
            }
            catch (Exception _)
            {
                try
                {
                    unit = (string)oldObject["base_unit"];
                }
                catch (Exception ex)
                {
                    //This DataType object won't work properly in ladybug python when serialised again as it requires either a base_unit or unit.
                    BH.Engine.Base.Compute.RecordError($"An error occurred when reading the unit of the DataType. returning unit as default ({unit}).\n The error: {ex}");
                }
            }

            if (dataType == "GenericType")
            {
                return new oM.LadybugTools.GenericDataType() { BaseUnit = unit, Data_Type = dataType, Name = name };
            }
            else
            {
                return new oM.LadybugTools.DataType() { Unit = unit, Data_Type = dataType, Name = name };
            }
            BH.Engine.Base.Compute.RecordError($"The data type being converted is borked.");
        }

        public static Dictionary<string, object> FromDataType(IDataType dataType)
        {
            Dictionary<string, object> returnDict = new Dictionary<string, object>();
            if (dataType.GetType().GetProperty("BaseUnit") != null)
            {
                returnDict.Add("base_unit", ((GenericDataType)dataType).BaseUnit);
            }
            else
            {
                returnDict.Add("unit", ((DataType)dataType).Unit);
            }

            returnDict.Add("name", ((DataType)dataType).Name);
            returnDict.Add("data_type", dataType.Data_Type);

            return returnDict;
        }
    }
}
