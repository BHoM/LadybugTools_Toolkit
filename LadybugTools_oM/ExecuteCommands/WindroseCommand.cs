using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that, when executed with the LadybugToolsAdapter, generates a windrose from the epw file using the given parameters.\nOutputs a string file path if the OutputLocation is given, or the base64 string representation of the image if no path is given.")]
    public class WindroseCommand : ISimulation
    {
        [Description("The path to an EPW file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The analysis period to use for plotting, default to whole non-leap year.")]
        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        [Description("The number of directional bins to plot on the windrose.")]
        public virtual int NumberOfDirectionBins { get; set; } = 36;

        [Description("A Matplotlib colour map. Corresponds to the 'cmap' parameter of plot methods. See https://matplotlib.org/stable/users/explain/colors/colormaps.html for examples of valid keys. Default of 'viridis'.")]
        public virtual string ColourMap { get; set; } = "viridis";

        [Description("Full file path (with file name) to save the plot to. Leave blank to output a base 64 string representation of the image instead.")]
        public virtual string OutputLocation { get; set; } = "";
    }
}