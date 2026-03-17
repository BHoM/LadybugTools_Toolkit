using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class CompareEPWKeyPlotCommand : ISimulationCommand
    {
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        public virtual List<FileSettings> EPWCompareFiles { get; set; } = new List<FileSettings>();

        public virtual bool PlotTimeseries { get; set; } = true;

        public virtual string OutputLocation { get; set; } = "";
    }
}
