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

using BH.oM.Base;
using BH.oM.Base.Attributes;
using BH.oM.Environment.Elements;
using System;
using System.ComponentModel;
using System.Threading;
using System.Linq;
using BH.oM.LadybugTools;
using BH.Engine.Geometry;
using BH.oM.Geometry;
using System.Runtime.CompilerServices;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts a Shelter object into a Panel.")]
        [Input("shelter", "Shelter object.")]
        [MultiOutput(0, "panel", "Environment panel representing the shelter object.")]
        [MultiOutput(1, "radiationPorosity", "The radiation porosity property of the shelter object, between 0 and 1.")]
        [MultiOutput(2, "windPorosity", "The wind porosity property of the shelter object, between 0 and 1.")]
        public static Output<Panel, double, double> FromShelter(this Shelter shelter)
        {
            if (shelter == null)
            {
                Base.Compute.RecordError("Shelter is null. Panel cannot be created.");
                return Base.Create.Output<Panel, double, double>(null, -1, -1);
            }

            Polyline pl = null;

            try
            {
                pl = Geometry.Create.Polyline(shelter.Vertices.Select(v => Geometry.Create.Point(v[0], v[1], v[2])).ToList());
            }
            catch (Exception ex)
            {
                Base.Compute.RecordError($"Error while trying to create panel from vertex data in shelter. Panel cannot be created.\nThe error was:\n{ex.Message}\n{ex.StackTrace}.");
                return Base.Create.Output<Panel, double, double>(null, -1, -1);
            }

            pl = pl.Close();

            if (!pl.IsPlanar())
            {
                Base.Compute.RecordError("Shelter is not planar, and so cannot be converted into a Panel.");
                return Base.Create.Output<Panel, double, double>(null, -1, -1);
            }

            return Base.Create.Output
                (
                    Environment.Create.Panel(Geometry.Create.PlanarSurface(pl)),
                    shelter.RadiationPorosity,
                    shelter.WindPorosity
                );
        }
    }
}

