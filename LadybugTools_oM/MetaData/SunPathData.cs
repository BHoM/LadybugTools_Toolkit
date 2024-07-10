using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class SunPathData : ISimulationData
    {
        public virtual SunData DecemberSolstice { get; set; } = new SunData();

        public virtual SunData MarchEquinox { get; set; } = new SunData();

        public virtual SunData JuneSolstice { get; set; } = new SunData();

        public virtual SunData SeptemberEquinox { get; set; } = new SunData();
    }
}
