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

using BH.Engine.Environment;
using BH.Engine.Geometry;
using BH.oM.Environment;
using BH.oM.Environment.Elements;
using BH.oM.Geometry;
using BH.oM.Reflection.Attributes;

using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace BH.Engine.LadybugTools
{
    public static partial class Create
    {
        [Description("Create a HB Face")]
        [Output("HBFace", "A Honeybee Face.")]
        public static HoneybeeSchema.Face CreateFace()
        {
            // Create some points defining vertices
            List<Point> vertices = new List<Point>
            {
                Geometry.Create.Point(0, 0, 0),
                Geometry.Create.Point(0, 1, 0),
                Geometry.Create.Point(1, 1, 0),
                Geometry.Create.Point(1, 0, 0)
            };
            List<List<double>> verticesList = new List<List<double>>();
            foreach (Point pt in vertices)
            {
                List<double> vertex = new List<double>
                {
                    pt.X,
                    pt.Y,
                    pt.Z
                };
                verticesList.Add(vertex);
            }

            HoneybeeSchema.FacePropertiesAbridged faceProperties = new HoneybeeSchema.FacePropertiesAbridged();
            HoneybeeSchema.Face face = new HoneybeeSchema.Face(
                identifier: "example_face", 
                geometry: new HoneybeeSchema.Face3D(verticesList), 
                faceType: HoneybeeSchema.FaceType.Wall, 
                boundaryCondition: new HoneybeeSchema.Outdoors(),
                properties: faceProperties);

            return face;
        }
    }

    public static partial class Convert
    {
        [Description("Convert a HB Face into a BHoM environment panel")]
        [Input("face", "A Honeybee Face object")]
        [Output("panel", "A BHoM environments panel.")]
        public static Panel ConvertFaceNET(HoneybeeSchema.Face face)
        {
            // Get geometry
            List<Point> vertices = new List<Point>();
            foreach (List<double> vertex in face.Geometry.Boundary)
            {
                vertices.Add(
                    Geometry.Create.Point(vertex[0], vertex[1], vertex[2])
                ) ;
            }
            Polyline panelPolyline = Geometry.Create.Polyline(vertices);
            

            // Create Panel
            Panel panel = new Panel();
            panel.ExternalEdges = panelPolyline.ToEdges();
            panel.Name = face.Identifier;
            panel.Type = PanelType.Wall;

            return panel;
        }

        [Description("Convert a HB face type into a BHoM Panel type")]
        [Input("faceType", "A Honeybee Face objects Type attribbute")]
        [Output("panelType", "A BHoM environments panel type enum.")]
        public static PanelType PanelTypeFromHoneybee(dynamic faceType)
        {
            switch (faceType.ToString())
            {
                case "Wall":
                    return PanelType.Wall;
                case "Floor":
                    return PanelType.Floor;
                case "RoofCeiling":
                    return PanelType.Roof;
                case "AirBoundary":
                    return PanelType.Air;
                default:
                    return PanelType.Wall;
            }
        }

        [Description("Convert a HB Face into set of edges")]
        [Input("face", "A Honeybee Face object")]
        [Output("edges", "A set of environment edges.")]
        public static List<Edge> EdgesFromHoneybee(dynamic face)
        {
            dynamic hbPoints = face._geometry._boundary;
            List<Point> vertices = new List<Point>();
            foreach (dynamic hbPoint in hbPoints)
            {
                vertices.Add(Geometry.Create.Point(hbPoint._x, hbPoint._y, hbPoint._z));
            }
            List<Edge> edges = Geometry.Create.Polyline(vertices).ToEdges();

            return edges;
        }

        [Description("Convert a HB Face into a BHoM environment panel")]
        [Input("face", "A Honeybee Face object")]
        [Output("panel", "A BHoM environments panel.")]
        public static Panel ConvertFacePython(dynamic face)
        {
            string identifier = face._identifier;
            string boundaryCondition = face._boundary_condition.ToString();
            PanelType type = PanelTypeFromHoneybee(face._type);
            List<Edge> edges = EdgesFromHoneybee(face);

            Panel panel = new Panel();
            panel.Name = identifier;
            panel.ExternalEdges = edges;
            panel.Type = type;

            return panel;
        }
    }
}
