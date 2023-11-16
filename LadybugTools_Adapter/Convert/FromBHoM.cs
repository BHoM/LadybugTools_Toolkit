using BH.oM.LadybugTools;
using System;
using BH.Engine.Serialiser;
using System.Collections.Generic;
using System.Text;
using BH.oM.Base;
using System.IO;
using BH.oM.Adapter;
using BH.Engine.Adapter;
using BH.oM.Base.Debugging;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static bool FromBHoM(this ILadybugTools input, FileSettings file)
        {
            string json = ICustomify(input);
            File.WriteAllText(file.GetFullFileName(), json);
            List<Event> events = BH.Engine.Base.Query.CurrentEvents();
            return events.Count == 0;
        }

        public static string ICustomify(this ILadybugTools LBTObject)
        {
            if (LBTObject == null)
            {
                BH.Engine.Base.Compute.RecordError("Input object is null.");
                return null;
            }
            return Jsonify(LBTObject as dynamic);
        }

        private static string Jsonify(this oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            
            return FromAnalysisPeriod(analysisPeriod).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.DataType dataType)
        {
            return FromDataType(dataType).ToJson();
        }

        private static string Jsonify(this oM.LadybugTools.HourlyContinuousCollection collection)
        {
            return FromHourlyContinuousCollection(collection);
        }

        private static string Jsonify(this oM.LadybugTools.Header header)
        {
            return FromHeader(header).ToJson();
        }

        private static Dictionary<string, object> Jsonify(this ILadybugTools obj)
        {
            BH.Engine.Base.Compute.RecordError($"The type: {obj.GetType()} is not convertible to ladybug serialisable json yet.");
            return null;
        }
    }
}
