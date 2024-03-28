using BH.oM.Adapter;
using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Reflection;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class DiurnalPlotCommand : ISimulationCommand
    {
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        public virtual EPWKey EPWKey { get; set; } = EPWKey.Undefined;

        public virtual Color Colour { get; set; }

        public virtual string OutputLocation { get; set; } = "";

        public virtual string Title { get; set; } = "";

        public virtual DiurnalPeriod Period { get; set; } = DiurnalPeriod.Undefined;

    }
}
