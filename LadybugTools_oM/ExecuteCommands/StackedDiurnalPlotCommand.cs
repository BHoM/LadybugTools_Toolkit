using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;

namespace BH.oM.LadybugTools.ExecuteCommands
{
    public class StackedDiurnalPlotCommand : ISimulationCommand
    {
        [Description("The EPW file to analyse.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The keys in the EPW file to analyse.")]
        public virtual List<EPWKey> EPWKeys { get; set; } = new List<EPWKey>();

        [Description("The colour of the average line on the plot.")]
        public virtual List<Color> Colours { get; set; } = new List<Color>();

        [Description("The directory to output the file. Leave empty to return a base64 string representation of that image.")]
        public virtual string OutputLocation { get; set; } = "";

        [Description("Title of the plot, will appear above any information on the top of the plot.")]
        public virtual string Title { get; set; } = "";

        [Description("The diurnal period to analyse. Daily for 365 samples/timestep, weekly for 52, monthly for 30.")]
        public virtual DiurnalPeriod Period { get; set; } = DiurnalPeriod.Undefined;
    }
}
