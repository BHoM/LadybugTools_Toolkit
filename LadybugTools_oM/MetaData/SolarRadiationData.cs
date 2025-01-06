using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [NoAutoConstructor]
    public class SolarRadiationData: ISimulationData
    {
        [Description("The maximum incoming solar radiation.")]
        public double MaxValue { get; set; } = double.NaN;

        [Description("The minimum incoming solar radiation.")]
        public double MinValue { get; set; } = double.NaN;

        [Description("The direction, in degrees(°) clockwise from north that the maximum incoming solar radiation is coming from.")]
        public double MaxDirection { get; set; } = double.NaN;

        [Description("The direction, in degrees(°) clockwise from north that the minimum incoming solar radiation is coming from.")]
        public double MinDirection { get; set; } = double.NaN;

        [Description("The angle, in degrees(°) above the horizon that the maximum incoming solar radiation is coming from.")]
        public double MaxTilt { get; set; } = double.NaN;

        [Description("The angle, in degrees(°) above the horizon that the minimum incoming solar radiation is coming from.")]
        public double MinTilt { get; set;} = double.NaN;
    }
}
