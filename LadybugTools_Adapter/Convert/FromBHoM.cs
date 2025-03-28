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

using BH.oM.LadybugTools;
using BH.Engine.Serialiser;
using System.Collections.Generic;
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

        private static string Jsonify(this AnalysisPeriod analysisPeriod)
        {
            return FromAnalysisPeriod(analysisPeriod).ToJson();
        }

        private static string Jsonify(this DataType dataType)
        {
            return FromDataType(dataType).ToJson();
        }

        private static string Jsonify(this EnergyMaterial energyMaterial)
        {
            return FromEnergyMaterial(energyMaterial).ToJson();
        }

        private static string Jsonify(this EnergyMaterialVegetation energyMaterial)
        {
            return FromEnergyMaterialVegetation(energyMaterial).ToJson();
        }

        private static string Jsonify(this ExternalComfort externalComfort)
        {
            return FromExternalComfort(externalComfort);
        }

        private static string Jsonify(this Header header)
        {
            return FromHeader(header).ToJson();
        }

        private static string Jsonify(this HourlyContinuousCollection collection)
        {
            return FromHourlyContinuousCollection(collection);
        }

        private static string Jsonify(this Shelter shelter)
        {
            return FromShelter(shelter);
        }

        private static string Jsonify(this SimulationResult simulationResult)
        {
            return FromSimulationResult(simulationResult);
        }

        private static string Jsonify(this Typology typology)
        {
            return FromTypology(typology);
        }

        private static Dictionary<string, object> Jsonify(this ILadybugTools obj)
        {
            BH.Engine.Base.Compute.RecordError($"The type: {obj.GetType()} is not convertible to ladybug serialisable json yet.");
            return null;
        }
    }
}


