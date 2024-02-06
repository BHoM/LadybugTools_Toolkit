using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using BH.oM.Adapter;

namespace BH.oM.LadybugTools
{
    public class RunUTCIHeatPlotCommand : IExecuteCommand
    {
        public virtual string EpwFile { get; set; } = "";
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; } = null;
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; } = null;
        public virtual Typology Typology { get; set; } = null;
        public virtual double WindSpeedMultiplier { get; set; } = 1;
        public virtual List<string> BinColours { get; set; } = new List<string>();
        public virtual string OutputLocation { get; set; } = "";
    }
}