using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class CollectionData : ISimulationData
    {
        public virtual double HighestValue { get; set; } = double.NaN;

        public virtual double LowestValue { get; set; } = double.NaN;

        public virtual DateTime HighestIndex { get; set; } = DateTime.MinValue;

        public virtual DateTime LowestIndex { get; set; } = DateTime.MinValue;

        public virtual double MedianValue { get; set; } = double.NaN;

        public virtual double MeanValue { get; set; } = double.NaN;

        public virtual List<double> MonthlyMeans { get; set; } = Enumerable.Repeat<double>(double.NaN, 12).ToList();

        public virtual List<List<double>> MonthlyDiurnalRanges { get; set; } = Enumerable.Repeat<List<double>>(new List<double> { double.NaN, double.NaN }, 12).ToList();
    }
}
