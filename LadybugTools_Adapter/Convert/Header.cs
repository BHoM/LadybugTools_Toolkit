using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        //TODO - find out if Header is needed or should just return an AnalysisPeriod
        public static BH.oM.LadybugTools.Header Header(BH.Adapter.LadybugTools.Header oldObject)
        {
            return new oM.LadybugTools.Header()
            {
                //AnalysisPeriod = AnalysisPeriod(oldObject.AnalysisPeriod),
            };
        }
    }
}
