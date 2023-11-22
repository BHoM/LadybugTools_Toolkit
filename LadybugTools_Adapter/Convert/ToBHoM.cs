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
using BH.Engine.Serialiser;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using BH.oM.Adapter;
using System.IO;
using BH.Engine.Adapter;
using BH.oM.Base;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ILadybugTools ToBHoM(this FileSettings jsonFile)
        {
            string json = File.ReadAllText(jsonFile.GetFullFileName());
            var obj = Engine.Serialiser.Convert.FromJson(json);

            Dictionary<string, object> LBTObject = null;

            if (obj.GetType() == typeof(CustomObject))
            {
                LBTObject = (obj as CustomObject).CustomData;
            }
            else if (obj.GetType() == typeof(Dictionary<string, object>))
            {
                BH.Engine.Base.Compute.RecordWarning("The object was not deserialised as a CustomObject, are you sure that this file came from a LadybugTools Python object? \n Trying to cast to Dictionary...");
                LBTObject = obj as Dictionary<string, object>;
            }
            else
            {
                BH.Engine.Base.Compute.RecordWarning($"The json given already deserialises to a BHoM object of type: {obj.GetType().FullName} - please use the BHoM Engine serialiser to deserialise this object.");
                return null;
            }

            switch (LBTObject["type"] as string)
            {
                case "AnalysisPeriod":
                    return ToAnalysisPeriod(LBTObject);
                case "GenericDataType":
                    return ToDataType(LBTObject);
                case "EnergyMaterial":
                    return ToEnergyMaterial(LBTObject);
                case "EnergyMaterialVegetation":
                    return ToEnergyMaterialVegetation(LBTObject);
                case "EPW":
                    return ToEPW(LBTObject);
                case "Header":
                    return ToHeader(LBTObject);
                case "HourlyContinuous":
                    return ToHourlyContinuousCollection(LBTObject);
                case "Location":
                    return ToLocation(LBTObject);
                default:
                    BH.Engine.Base.Compute.RecordError("The json file given is not convertable to a LadybugTools object.");
                    return null;
            }
        }
    }
}