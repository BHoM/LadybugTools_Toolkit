/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2025, the respective contributors. All rights reserved.
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
        //TODO - find out if Header is needed or should just return an AnalysisPeriod
        public static BH.oM.LadybugTools.Header ToHeader(Dictionary<string, object> oldObject)
        {
            string unit = "";
            DataType dataType = new DataType();
            AnalysisPeriod analysisPeriod = new AnalysisPeriod();
            Dictionary<string, object> metaData = new Dictionary<string, object>();

            try
            {
                if (oldObject["data_type"].GetType() == typeof(CustomObject))
                    oldObject["data_type"] = (oldObject["data_type"] as CustomObject).CustomData;
                dataType = ToDataType(oldObject["data_type"] as Dictionary<string, object>);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the DataType of the Header. returning a default DataType object instead.\n The error: {ex}");
            }

            try
            {
                if (oldObject["analysis_period"].GetType() == typeof(CustomObject))
                    oldObject["analysis_period"] = (oldObject["analysis_period"] as CustomObject).CustomData;
                analysisPeriod = ToAnalysisPeriod(oldObject["analysis_period"] as Dictionary<string, object>);
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the AnalysisPeriod of the Header. returning a default AnalysisPeriod.\n The error: {ex}");
            }

            try
            {
                if (oldObject["metadata"].GetType() == typeof(CustomObject))
                    oldObject["metadata"] = (oldObject["metadata"] as CustomObject).CustomData;
                metaData = (Dictionary<string, object>)oldObject["metadata"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the meta data of the Header. returning an empty metadata object.\n The error: {ex}");
            }

            try
            {
                unit = (string)oldObject["unit"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the unit of the Header. returning unit as default (\"\").\n The error: {ex}");
            }

            return new Header()
            {
                Unit = unit,
                DataType = dataType,
                AnalysisPeriod = analysisPeriod,
                Metadata = metaData
            };
        }

        public static Dictionary<string, object> FromHeader(BH.oM.LadybugTools.Header header)
        {
            return new Dictionary<string, object>()
            {
                { "type", "Header" },
                { "data_type", FromDataType(header.DataType) },
                { "unit", header.Unit },
                { "analysis_period", FromAnalysisPeriod(header.AnalysisPeriod) },
                { "metadata", header.Metadata }
            };
        }
    }
}


