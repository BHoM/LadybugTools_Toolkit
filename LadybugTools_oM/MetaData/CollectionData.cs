using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class CollectionData : ISimulationData
    {
        public virtual List<string> Description { get; set; } = new List<string>();

        public virtual double HighestValue { get; set; } = double.NaN;

        public virtual double LowestValue { get; set; } = double.NaN;

        public virtual DateTime HighestTime { get; set; } = DateTime.MinValue;

        public virtual DateTime LowestTime { get; set; } = DateTime.MinValue;

        public virtual double HighestAverageMonthValue { get; set; } = double.NaN;

        public virtual double LowestAverageMonthValue { get; set; } = double.NaN;

        public virtual int HighestAverageMonth { get; set; } = 0;

        public virtual int LowestAverageMonth { get; set; } = 0;
    }
}
