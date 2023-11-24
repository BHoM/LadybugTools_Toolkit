using System;
using System.Collections.Generic;
using System.Text;
using BH.oM.LadybugTools;
using BH.oM.Geometry;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static Point ToPoint(Dictionary<string, object> oldObject)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;

            try
            {
                x = (double)oldObject["x"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading x of the Point. returning x as default ({x}).\n The error: {ex}");
            }

            try
            {
                y = (double)oldObject["y"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading y of the Point. returning y as default ({y}).\n The error: {ex}");
            }

            try
            {
                z = (double)oldObject["z"];
            }
            catch (Exception ex)
            {
                BH.Engine.Base.Compute.RecordError($"An error occurred when reading z of the Point. returning z as default ({z}).\n The error: {ex}");
            }

            return new Point()
            {
                X = x,
                Y = y,
                Z = z
            };
        }

        public static string FromPoint(Point point)
        {
            string type = @"""Point3D""";
            string xyz = $@"""x"" : {point.X}, ""y"" : {point.Y}, ""z"" : {point.Z}";
            return @"{""type"" : " + type + ", " + xyz + "}";
        }
    }
}
