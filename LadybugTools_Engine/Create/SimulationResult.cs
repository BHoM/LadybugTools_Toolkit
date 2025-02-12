using BH.oM.Adapter;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Engine.LadyBugTools
{
    public static partial class Create
    {
        public static SimulationResult SimulationResult(FileSettings epwFile, string identifier, IEnergyMaterialOpaque groundMaterial, IEnergyMaterialOpaque shadeMaterial)
        {
            return new SimulationResult()
            {
                EpwFile = epwFile,
                Name = identifier,
                GroundMaterial = groundMaterial,
                ShadeMaterial = shadeMaterial
            };
        }
    }
}
