using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class SunData
    {
        public virtual double SunriseAzimuth { get; set; } = double.NaN;

        public virtual DateTime SunriseTime { get; set; } = DateTime.MinValue;

        public virtual double NoonAltitude { get; set; } = double.NaN;

        public virtual DateTime NoonTime { get; set; } = DateTime.MinValue;

        public virtual double SunsetAzimuth { get; set; } = double.NaN;

        public virtual DateTime SunsetTime { get; set; } = DateTime.MinValue;
    }
}
