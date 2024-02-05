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
        public virtual EnergyMaterial GroundMaterial { get; set; } = null;
        public virtual EnergyMaterial ShadeMaterial { get; set; } = null;
        public virtual Typology Typology { get; set; } = null;
        public virtual double EvaporativeCooling { get; set; } = 0;
        public virtual double WindSpeedMultiplier { get; set; } = 1;
        public virtual List<Color> BinColours { get; set; } = new List<Color>();
        public virtual string OutputLocation { get; set; } = "";
    }
}