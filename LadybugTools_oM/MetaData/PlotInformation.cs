using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class PlotInformation : BHoMObject
    {
        public virtual string Image { get; set; } = "";

        public virtual ISimulationData OtherData { get; set; }
    }
}
