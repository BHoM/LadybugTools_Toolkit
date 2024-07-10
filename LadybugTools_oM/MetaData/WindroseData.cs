using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class WindroseData : ISimulationData
    {
        public virtual List<double> PrevailingDirection { get; set; } = Enumerable.Repeat<double>(double.NaN, 2).ToList();

        public virtual double PrevailingPercentile95 { get; set; } = double.NaN;

        public virtual double PrevailingPercentile50 { get; set; } = double.NaN;

        public virtual double Percentile95 { get; set; } = double.NaN; 

        public virtual double Percentile50 { get; set; } = double.NaN;

        public virtual double PercentageOfCalmHours { get; set; } = double.NaN;
    }
}
