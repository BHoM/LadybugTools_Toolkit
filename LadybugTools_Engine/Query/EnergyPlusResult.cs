/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2021, the respective contributors. All rights reserved.
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
using LadybugTools_oM.Enums;
using BH.oM.Reflection.Attributes;

using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Data.SQLite;
using System.Reflection;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Create an EnergyPlusResult object from a SQLite file containing Ladybug HourlyContinuousCollections.")]
        [Input("sqliteFile", "A SQLite containing Ladybug HourlyContinuousCollections.")]
        [Input("energyPlusResultType", "A result attribute to query from the SQLite simulation results.")]
        [Output("energyPlusResult", "An EnergyPlusResult object.")]
        public static EnergyPlusResult EnergyPlusResult(string sqliteFile, EnergyPlusResultType energyPlusResultType = EnergyPlusResultType.Undefined)
        {
            // TODO - Add return null if energyPlusResultType is null

            EnergyPlusResult energyPlusResult = new EnergyPlusResult
            {
                Name = Path.GetFileNameWithoutExtension(sqliteFile)
                // TODO - get the proper name for the simulation from the SQL file itself!
            };

            // Connect to the SQL file
            SQLiteConnection con = new SQLiteConnection(sqliteFile);
            con.Open();

            string query = string.Format("SELECT ReportDataDictionaryIndex, IndexGroup, KeyValue, Name, Units FROM ReportDataDictionary WHERE Name = {0}", GetEnumDescription(energyPlusResultType));
            SQLiteCommand cmd = new SQLiteCommand(query, con);
            SQLiteDataReader rdr = cmd.ExecuteReader();
            while (rdr.Read())
            {
                energyPlusResult.Results.Add("");
            }

            


            return energyPlusResult;
        }

        private static string GetEnumDescription(System.Enum value)
        {
            FieldInfo fi = value.GetType().GetField(value.ToString());

            DescriptionAttribute[] attributes = fi.GetCustomAttributes(typeof(DescriptionAttribute), false) as DescriptionAttribute[];

            if (attributes != null && attributes.Any())
            {
                return attributes.First().Description;
            }

            return value.ToString();
        }
    }
}