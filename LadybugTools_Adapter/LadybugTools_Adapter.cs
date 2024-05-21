/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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

using System.ComponentModel;
using BH.Engine.Python;
using System.IO;
using BH.oM.Base.Attributes;
using BH.oM.Python;

namespace BH.Adapter.LadybugTools
{
    public partial class LadybugToolsAdapter : BHoMAdapter
    {
        [Description("Produces a LadybugTools Adapter that converts objects between Ladybug compatible json and BHoM objects.")]
        [Output("adapter", "Adapter to a LadybugTools object.")]
        public LadybugToolsAdapter()
        {
            m_AdapterSettings.DefaultPushType = oM.Adapter.PushType.CreateOnly;

            //get the base python environment first, as LBT is dependant on it.
            BH.Engine.Python.Compute.BasePythonEnvironment(run: true);
            m_environment = BH.Engine.LadybugTools.Compute.InstallPythonEnv_LBT(run: true);
        }

        public LadybugToolsAdapter(PythonEnvironment environment)
        {
            m_AdapterSettings.DefaultPushType = oM.Adapter.PushType.CreateOnly;
            m_environment = environment;
        }

        private readonly PythonEnvironment m_environment;
    }
}