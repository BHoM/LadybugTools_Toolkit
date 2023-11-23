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
using System.Text;
using BH.oM.Base;
using System.IO;
using BH.oM.Adapter;
using BH.Engine.Adapter;
using BH.oM.Base.Debugging;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static string FromBHoM(this ILadybugTools input)
        {
            return ICustomify(input);
        }

        public static string ICustomify(this ILadybugTools lbtObject)
        {
            if (lbtObject == null)
            {
                BH.Engine.Base.Compute.RecordError("Input object is null.");
                return null;
            }
            return Jsonify(lbtObject as dynamic);
        }

        private static string Jsonify(this oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            return FromAnalysisPeriod(analysisPeriod).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.DataType dataType)
        {
            return FromDataType(dataType).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.EnergyMaterial energyMaterial)
        {
            return FromEnergyMaterial(energyMaterial).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.EnergyMaterialVegetation energyMaterial)
        {
            return FromEnergyMaterialVegetation(energyMaterial).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.EPW epw)
        {
            return FromEPW(epw);
        }

        private static string Jsonify(this oM.LadybugTools.Header header)
        {
            return FromHeader(header).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.HourlyContinuousCollection collection)
        {
            return FromHourlyContinuousCollection(collection);
        }

        private static string Jsonify(this oM.LadybugTools.Location location)
        {
            return FromLocation(location).ToJson();
        }

        private static Dictionary<string, object> Jsonify(this ILadybugTools obj)
        {
            BH.Engine.Base.Compute.RecordError($"The type: {obj.GetType()} is not convertible to ladybug serialisable json yet.");
            return null;
        }
    }
}
