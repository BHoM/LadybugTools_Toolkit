/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2020, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using BH.oM.Geometry;
using BH.oM.Environment;
using BH.oM.Environment.Elements;

using BH.Engine.Geometry;
using BH.Engine.Environment;

using BH.oM.Reflection.Attributes;
using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Convert
    {
        [Description("Converts a Honeybee Surface object into an Environments Panel or Opening depending on the Honeybee Surface type.\n" +
            "If the Honeybee Surface is a Window type, then it will convert to an Environments Opening, otherwise it will convert to an Environments Panel. If the Honeybee Surface is an AirWall type then it will convert to an Environments Panel which contains an Opening with the same edges, and the Opening will be set as the Hole type.\n" +
            "Providing a collection of Honeybee Surfaces will return a collection of Environment objects, which may be mixed as Panels and Openings depending on the types.\n" +
            "You can filter out Panels from Openings using the Panels(List<IBHoMObject>) and Openings(List<IBHoMObject>) query methods in the Environment Engine.")]
        [Input("honeybeeSrf", "A Honeybee Surface object from the Honeybee Plus libraries")]
        [Output("environmentObject", "Either an Environment Panel or an Environment Opening depending on the Honeybee Surface type.")]
        public static IEnvironmentObject FromHoneybeeSurface(dynamic honeybeeSrf)
        {
            IGeometry geometry = null;
            try
            {
                geometry = BH.Engine.Rhinoceros.Convert.FromRhino(honeybeeSrf._geometry as dynamic);
            }
            catch(Exception e)
            {
                BH.Engine.Reflection.Compute.RecordError("Could not convert that Honeybee Surface to an Environments Panel - recorded error was: " + e.ToString());
                return null;
            }

            if(geometry == null)
            {
                BH.Engine.Reflection.Compute.RecordError("Honeybee Surface does not contain any valid geometry");
                return null;
            }

            Polyline boundary = null;
            try
            {
                if (geometry is ISurface)
                    boundary = (geometry as ISurface).IExternalEdges().ToList().IJoin().FirstOrDefault().CollapseToPolyline(BH.oM.Geometry.Tolerance.Angle);
                else if (geometry is ICurve)
                    boundary = (geometry as ICurve).ICollapseToPolyline(BH.oM.Geometry.Tolerance.Angle);
            }
            catch(Exception e)
            {
                BH.Engine.Reflection.Compute.RecordError("Honeybee Surface geometry could not be converted successfully - recorded error was: " + e.ToString());
                return null;
            }

            if(boundary == null)
            {
                BH.Engine.Reflection.Compute.RecordError("Honeybee Surface geometry could not be converted to BHoM Geometry");
                return null;
            }

            string panelType = string.Empty;
            try
            {
                panelType = honeybeeSrf._surface_type.ToString();//.__class__.__doc__;
            }
            catch(Exception e)
            {
                BH.Engine.Reflection.Compute.RecordWarning("Honeybee Surface does not contain a valid panel type. Using PanelType.ExternalWall instead. Recorded error was: " + e.ToString());
            }

            if(panelType == string.Empty)
                BH.Engine.Reflection.Compute.RecordWarning("Honeybee Surface does not contain a valid panel type. Using PanelType.ExternalWall instead");

            if (panelType.EndsWith("Window"))
                return ToOpening(boundary.ToEdges());
            else
                return ToPanel(boundary.ToEdges(), panelType);            
        }

        private static Panel ToPanel(List<Edge> edges, string panelType)
        {
            Panel returnPanel = new Panel();
            returnPanel.ExternalEdges = edges;

            PanelType type = PanelType.WallExternal;
            if (panelType.EndsWith("Wall"))
                type = PanelType.WallExternal;
            else if (panelType.EndsWith("Roof"))
                type = PanelType.Roof;
            else if (panelType.EndsWith("Floor"))
                type = PanelType.Floor;
            else if (panelType.EndsWith("Ceiling"))
                type = PanelType.Ceiling;

            returnPanel.Type = type;

            if (panelType.EndsWith("AirWall"))
            {
                //This should have a 100% opening on the wall
                Opening airWall = new Opening();
                airWall.Type = OpeningType.Undefined; //Change to Hole when implemented
                airWall.Edges = edges;
                returnPanel.Openings.Add(airWall);
            }

            return returnPanel;
        }

        private static Opening ToOpening(List<Edge> edges)
        {
            Opening returnOpening = new Opening();
            returnOpening.Edges = edges;
            returnOpening.Type = OpeningType.Window;

            return returnOpening;
        }
    }
}
