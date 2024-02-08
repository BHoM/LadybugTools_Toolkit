using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that, when executed with the LadybugToolsAdapter, produces a heatmap from data in an epw file.\nOutput is a string of either the path to the image (if OutputLocation is not set) or the base 64 string representation of it.")]
    public class HeatPlotCommand : ISimulation
    {
        [Description("The path to an EPW file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("Key representing an hourly continuous collection in the EPW file to plot.")]
        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        [Description("A Matplotlib colour map. Corresponds to the 'cmap' parameter of plot methods. See https://matplotlib.org/stable/users/explain/colors/colormaps.html for examples of valid keys. Default of 'viridis'.")]
        public virtual string ColourMap { get; set; } = "viridis";

        [Description("Full file path (with file name) to save the plot to. Leave blank to output a base 64 string representation of the image instead.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}