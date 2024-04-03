using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class SunPathPlotCommand : ISimulationCommand
    {
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        public virtual double SunSize { get; set; } = 10;

        public virtual string OutputLocation { get; set; } = "";
    }
}
