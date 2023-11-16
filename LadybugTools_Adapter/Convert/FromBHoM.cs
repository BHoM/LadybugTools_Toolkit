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
        public static bool FromBHoM(this ILadybugTools input, FileSettings json)
        {
            Dictionary<string, object> obj = ICustomify(input);
            File.WriteAllText(json.GetFullFileName(), obj.ToJson());
            List<Event> events = BH.Engine.Base.Query.CurrentEvents();
            return events.Count == 0;
        }

        public static Dictionary<string, object> ICustomify(this ILadybugTools LBTObject)
        {
            if (LBTObject == null)
            {
                BH.Engine.Base.Compute.RecordError("Input object is null.");
                return null;
            }
            return Dictify(LBTObject as dynamic);
        }

        private static Dictionary<string, object> Dictify(this oM.LadybugTools.AnalysisPeriod analysisPeriod)
        {
            return FromAnalysisPeriod(analysisPeriod);
        }

        private static Dictionary<string, object> Dictify(this oM.LadybugTools.DataType dataType)
        {
            return FromDataType(dataType);
        }
    }
}
