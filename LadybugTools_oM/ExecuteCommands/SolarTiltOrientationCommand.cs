using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class SolarTiltOrientationCommand : ISimulationCommand
    {
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        public virtual int Azimuths { get; set; } = 3;

        public virtual int Altitudes { get; set; } = 3;

        public virtual double GroundReflectance { get; set; } = 0.2;

        public virtual IrradianceType IrradianceType { get; set; } = IrradianceType.Total;

        public virtual bool Isotropic { get; set; } = true;

        public virtual AnalysisPeriod AnalysisPeriod { get; set; } = new AnalysisPeriod();

        public virtual string Title { get; set; } = "";

        public virtual string OutputLocation { get; set; } = "";
    }
}