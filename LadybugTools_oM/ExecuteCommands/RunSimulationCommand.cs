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

using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that when executed with the LadybugTools Adapter, runs a simulation and return a SimulationResult containing hourly data.")]
    public class RunSimulationCommand : ISimulationCommand, IObject
    {
        [Description("FileSettings for an EPW file to run the simulation with.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The ground material for the simulation to use.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; } = null;

        [Description("The shade material for the simulation to use.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; } = null;
    }
}
