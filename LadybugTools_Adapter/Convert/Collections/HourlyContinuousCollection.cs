using BH.Engine.Serialiser;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static BH.oM.LadybugTools.HourlyContinuousCollection ToHourlyContinuousCollection(Dictionary<string, object> oldObject)
        {
            return new oM.LadybugTools.HourlyContinuousCollection()
            {
                Values = (List<double>)oldObject["Values"],
            };
        }

        public static string FromHourlyContinuousCollection(BH.oM.LadybugTools.HourlyContinuousCollection collection)
        {
            string type = "\"type\" : \"HourlyContinuous\"";
            string values = "\"values\" : [" + string.Join(", ", collection.Values.Select(x => x.ToString()).ToList()) + " ]";
            string header = "\"header\" : " + ICustomify(collection.Header);
            return "{ " + type + "," + values + "," + header + " }";
        }
    }
}
