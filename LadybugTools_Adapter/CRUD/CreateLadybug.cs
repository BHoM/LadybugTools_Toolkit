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

using BH.Engine.Adapter;
using BH.oM.Adapter;
using BH.oM.Base.Debugging;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public static void CreateLadybug(List<ILadybugTools> objects, LadybugConfig config = null)
        {
            List<string> jsonObjects = new List<string>();
            
            foreach (ILadybugTools lbtObject in objects)
            {
                jsonObjects.Add(lbtObject.FromBHoM());
            }
            string json = "{}";
            if (jsonObjects.Count > 1)
                json = $"[{string.Join(", ", jsonObjects)}]";
            else if (jsonObjects.Count == 1)
                json = jsonObjects[0];
            File.WriteAllText(config.JsonFile.GetFullFileName(), json);
        }
    }
}


