using BH.oM.LadybugTools;
using System;
using BH.Engine.Serialiser;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadyBugTools
{
    public static partial class Convert
    {
        public static ILadybugTools FromBHoM(this string json)
        {
            ILadybugTools converted = Engine.Serialiser.Convert.FromJson(json) as ILadybugTools;
            return converted;
        }
    }
}
