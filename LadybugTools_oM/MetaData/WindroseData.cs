using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class WindroseData : ISimulationData
    {
        public virtual List<string> Description { get; set; } = new List<string>();

        public virtual double PrevailingDirection { get; set; } = double.NaN;

        public virtual double PrevailingSpeed { get; set; } = 0;

        public virtual DateTime PrevailingTime {  get; set; } = DateTime.MinValue;

        public virtual double MaxSpeed { get; set; } = 0;

        public virtual double MaxSpeedDirection { get; set; } = double.NaN;

        public virtual DateTime MaxSpeedTime { get; set; } = DateTime.MinValue;

        public virtual double NumberOfCalmHours { get; set; } = 0;

        public virtual double PercentageOfCalmHours { get; set; } = 0;
    }
}
