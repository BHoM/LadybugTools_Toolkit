using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class RunHeatPlotCommand : IExecuteCommand
    {
        public virtual string EpwFile { get; set; } = "";
        public virtual EpwKey EpwKey { get; set; } = EpwKey.Undefined;
        public virtual string ColourMap { get; set; } = "viridis";
        public virtual string OutputLocation { get; set; } = "";
    }
}
