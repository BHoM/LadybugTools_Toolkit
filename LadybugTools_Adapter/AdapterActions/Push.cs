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
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override List<object> Push(IEnumerable<object> objects, string tag = "", PushType pushType = PushType.AdapterDefault, ActionConfig actionConfig = null)
        {
            if (actionConfig == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid LadybugConfig ActionConfig.");
                return new List<object>();
            }

            LadybugConfig config = actionConfig as LadybugConfig;
            if (config == null)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid LadybugConfig.");
                return new List<object>();
            }

            if (objects.Count() == 0)
            {
                BH.Engine.Base.Compute.RecordError("Please input a valid LadybugTools object.");
                return new List<object>();
            }

            List<ILadybugTools> lbtObjects = objects.Where(x => typeof(ILadybugTools).IsAssignableFrom(x.GetType())).Cast<ILadybugTools>().ToList();

            if (lbtObjects.Count() < objects.Count())
            {
                BH.Engine.Base.Compute.RecordWarning("The LadybugTools Toolkit adapter does not support converting non-ILadybugTools objects to json, skipping all objects that are not an ILadybugTools");
            }

            CreateLadybug(lbtObjects, config);
            return objects.Where(x => typeof(ILadybugTools).IsAssignableFrom(x.GetType())).ToList();
        }
    }
}


