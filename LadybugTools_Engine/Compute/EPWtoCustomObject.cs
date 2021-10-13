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

using BH.oM.Base;
using BH.oM.Reflection.Attributes;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;

namespace BH.Engine.LadybugTools
{
    public static partial class Compute
    {
        [Description("Convert an EPW file into a BHoM CustomObject.")]
        [Input("epwFile", "An EPW file.")]
        [Output("customObject", "A BHoM CustomObject.")]
        public static CustomObject EPWtoCustomObject(string epwFile)
        {
            string scriptPath = @"C:\ProgramData\BHoM\Extensions\LadybugTools\EPWtoJSON.py";
            string output = Python.Compute.RunCommand(VIRTUALENV_NAME, scriptPath, new List<string>() { epwFile });

            // Replace "Infinity" values in JSON to avoid issues with Serialiser.Engine
            output = output.Trim().Replace("Infinity", "0");

            //return output;
            // convert output into a CustomObject
            // TODO - Convert CustomObject into a BHoM serialisable object to enable bi-directional conversion
            return Serialiser.Convert.FromJson(output) as CustomObject;
        }
    }
}
