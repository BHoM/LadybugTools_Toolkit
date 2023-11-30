using BH.oM.Adapter;
using BH.oM.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that when executed with the LadybugTools Adapter, runs a simulation and return a SimulationResult containing hourly data.")]
    public class RunSimulationCommand : IExecuteCommand, IObject
    {
        [Description("FileSettings for an EPW file to run the simulation with.")]
        public virtual FileSettings EpwFile { get; set; } = new FileSettings();

        [Description("The ground material for the simulation to use.")]
        public virtual IEnergyMaterialOpaque GroundMaterial { get; set; } = null;

        [Description("The shade material for the simulation to use.")]
        public virtual IEnergyMaterialOpaque ShadeMaterial { get; set; } = null;
    }
}