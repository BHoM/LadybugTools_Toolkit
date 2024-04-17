using BH.oM.Adapter;
using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Reflection;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Use in conjunction with the LadybugToolsAdapter to run a diurnal analysis on a specific key of an epw file, and output a plot.")]
    public class DiurnalPlotCommand : ISimulationCommand
    {
        [Description("The EPW file to analyse.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The key in the EPW file to analyse.")]
        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        [Description("The colour of the average line on the plot.")]
        public virtual Color Colour { get; set; }

        [Description("The directory to output the file. Leave empty to return a base64 string representation of that image.")]
        public virtual string OutputLocation { get; set; } = "";

        [Description("Title of the plot, will appear above any information on the top of the plot.")]
        public virtual string Title { get; set; } = "";

        [Description("The diurnal period to analyse. Daily for 365 samples/timestep, weekly for 52, monthly for 30.")]
        public virtual DiurnalPeriod Period { get; set; } = DiurnalPeriod.Undefined;
    }
}
