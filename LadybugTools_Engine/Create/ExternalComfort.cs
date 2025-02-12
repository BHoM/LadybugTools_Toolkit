using BH.Engine.Adapter;
using BH.Engine.LadybugTools;
using BH.oM.Adapter;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Engine.LadyBugTools
{
    public static partial class Create
    {
        public static ExternalComfort ExternalComfort(FileSettings epwFile, string identifier, IEnergyMaterialOpaque groundMaterial, IEnergyMaterialOpaque shadeMaterial, Typology typology)
        {
            return new ExternalComfort()
            {
                SimulationResult = SimulationResult(epwFile, identifier, groundMaterial, shadeMaterial),
                Typology = typology
            };
        }

        public static ExternalComfort ExternalComfort(SimulationResult simulationResult, Typology typology)
        {
            return new ExternalComfort()
            {
                SimulationResult = simulationResult,
                Typology = typology
            };
        }
    }
}
