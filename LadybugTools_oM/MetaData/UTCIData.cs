using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class UTCIData : ISimulationData
    {
        public virtual double ComfortableRatio { get; set; } = double.NaN;

        public virtual double HotRatio { get; set; } = double.NaN;

        public virtual double ColdRatio { get; set; } = double.NaN;

        public virtual double DaytimeComfortableRatio { get; set; } = double.NaN;

        public virtual double DaytimeHotRatio { get; set; } = double.NaN;

        public virtual double DaytimeColdRatio { get; set; } = double.NaN;
    }
}
