using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using BH.Engine.Serialiser;
using BH.oM.Base;
using BH.oM.Geometry;
using BH.oM.LadybugTools;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static oM.LadybugTools.Shelter ToShelter(Dictionary<string, object> oldObject)
        {
            List<Point> points = new List<Point>();
            List<double> radiationPorosity = Enumerable.Repeat(0.0, 8760).ToList();
            List<double> windPorosity = Enumerable.Repeat(0.0, 8760).ToList();
            if (oldObject.ContainsKey("vertices"))
            {
                
                foreach (var vertex in oldObject["vertices"] as List<object>)
                {
                    Dictionary<string, object> point;
                    if (vertex.GetType() == typeof(CustomObject))
                    {
                        point = (vertex as CustomObject).CustomData;
                    }
                    else
                    {
                        point = vertex as Dictionary<string, object>;
                    }

                    points.Add(ToPoint(point));
                }
            }

            try
            {
                List<double> values = new List<double>();
                foreach (object value in oldObject["radiation_porosity"] as List<object>)
                {
                    values.Add(double.Parse(value.ToString()));
                }
                radiationPorosity = values;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the radiation porosity of the Shelter. returning as default (List of 0s of length 8760).\n The error: {ex}");
            }

            try
            {
                List<double> values = new List<double>();
                foreach (object value in oldObject["wind_porosity"] as List<object>)
                {
                    values.Add(double.Parse(value.ToString()));
                }
                windPorosity = values;
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading the wind porosity of the Shelter. returning as default (List of 0s of length 8760).\n The error: {ex}");
            }

            return new oM.LadybugTools.Shelter()
            {
                Vertices = points,
                RadiationPorosity = radiationPorosity,
                WindPorosity = windPorosity
                
            };
        }

        public static string FromShelter(oM.LadybugTools.Shelter shelter)
        {
            string radiationPorosity = $@"""radiation_porosity"": [{string.Join(", ", shelter.RadiationPorosity)}]";
            string windPorosity = $@"""wind_porosity"": [{string.Join(", ", shelter.WindPorosity)}]";

            List<string> points = new List<string>();
            foreach (Point point in shelter.Vertices)
            {
                points.Add(FromPoint(point));
            }
            string vertices = $@"""vertices"": [{string.Join(", ", points)}]";

            return @"{ ""type"": ""Shelter""," + vertices + ", " + radiationPorosity + ", " + windPorosity + "}";
        }
    }
}
