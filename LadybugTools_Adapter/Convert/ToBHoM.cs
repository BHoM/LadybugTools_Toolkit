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
        public static List<IBHoMObject> ToBHoM(this FileSettings jsonFile)
        {
            string json = File.ReadAllText(jsonFile.GetFullFileName());
            if (!json.StartsWith("["))
                json = "[" + json;
                
            if (!json.EndsWith("]"))
                json = json + "]";
            IEnumerable<object> objs = Engine.Serialiser.Convert.FromJsonArray(json);
            List<IBHoMObject> returnObjects = new List<IBHoMObject>();
            foreach (var obj in objs)
            {
                Dictionary<string, object> lbtObject = null;
                if (obj.GetType() == typeof(CustomObject))
                {
                    lbtObject = (obj as CustomObject).CustomData;
                }
                else if (obj.GetType() == typeof(Dictionary<string, object>))
                {
                    BH.Engine.Base.Compute.RecordWarning("The object was not deserialised as a CustomObject, are you sure that this file came from a LadybugTools Python object? \n Trying to cast to Dictionary...");
                    lbtObject = obj as Dictionary<string, object>;
                }
                else
                {
                    BH.Engine.Base.Compute.RecordWarning($"One of the objects in the json given already deserialises to a BHoM object of type: {obj.GetType().FullName}. Returning this object.");
                    returnObjects.Add((IBHoMObject)obj);
                    continue;
                }
                if (lbtObject.ContainsKey("type"))
                {
                    switch (lbtObject["type"] as string)
                    {
                        case "AnalysisPeriod":
                            returnObjects.Add(ToAnalysisPeriod(lbtObject));
                            break;
                        case "GenericDataType":
                            returnObjects.Add(ToDataType(lbtObject));
                            break;
                        case "DataType":
                            returnObjects.Add(ToDataType(lbtObject));
                            break;
                        case "EnergyMaterial":
                            returnObjects.Add(ToEnergyMaterial(lbtObject));
                            break;
                        case "EnergyMaterialVegetation":
                            returnObjects.Add(ToEnergyMaterialVegetation(lbtObject));
                            break;
                        case "EPW":
                            returnObjects.Add(ToEPW(lbtObject));
                            break;
                        case "Header":
                            returnObjects.Add(ToHeader(lbtObject));
                            break;
                        case "HourlyContinuous":
                            returnObjects.Add(ToHourlyContinuousCollection(lbtObject));
                            break;
                        case "Location":
                            returnObjects.Add(ToLocation(lbtObject));
                            break;
                        case "Shelter":
                            returnObjects.Add(ToShelter(lbtObject));
                            break;
                        case "Typology":
                            returnObjects.Add(ToTypology(lbtObject));
                            break;
                        default:
                            BH.Engine.Base.Compute.RecordError($"Objects of type {lbtObject["type"]} are not yet supported for conversion to a LadybugTools object.");
                            break;
                    }
                }
                else
                {
                    BH.Engine.Base.Compute.RecordError("One of the objects in the json file given does not specify the type of the object contained.");
                    return null;
                }
            }
            return returnObjects;
        }
    }
}