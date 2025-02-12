using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class WalkabilityPlotCommand: ISimulationCommand
    {
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        public virtual ExternalComfort ExternalComfort { get; set; } = null;

        public virtual string OutputLocation { get; set; } = "";
    }
}
