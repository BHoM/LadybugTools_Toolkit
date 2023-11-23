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

using BH.Engine.Adapter;
using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.Data.Requests;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        public override IEnumerable<object> Pull(IRequest request, PullType pullType = PullType.AdapterDefault, ActionConfig actionConfig = null) 
        {
            LadybugConfig config = actionConfig as LadybugConfig;
            if (config == null)
            {
                BH.Engine.Base.Compute.RecordError($"The type of actionConfig provided: {actionConfig.GetType().FullName} is not valid for this adapter. Please provide a valid LadybugConfig actionConfig.");
                return new List<IBHoMObject>();
            }

            if (config.JsonFile == null)
            {
                BH.Engine.Base.Compute.RecordError("Please provide a valid JsonFile FileSettings object.");
                return new List<IBHoMObject>();
            }

            if (!System.IO.File.Exists(config.JsonFile.GetFullFileName()))
            {
                BH.Engine.Base.Compute.RecordError($"The file at {config.JsonFile.GetFullFileName()} does not exist to pull from.");
                return new List<IBHoMObject>();
            }

            if (request != null)
            {
                FilterRequest filterRequest = request as FilterRequest;
                return Read(filterRequest.Type, actionConfig: config);
            }
            else
                return Read(null, config);
        }
    }
}
