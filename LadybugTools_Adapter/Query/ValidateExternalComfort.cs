using BH.Engine.Base;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Query
    {
        public static bool ValidateExternalComfort(ExternalComfort externalComfort)
        {
            if (externalComfort.UniversalThermalClimateIndex?.Values.IsNullOrEmpty() ?? false)
            {
                if (externalComfort.SimulationResult.GroundMaterial == null)
                {
                    BH.Engine.Base.Compute.RecordError($"Please provide a valid ground material to the simulation result in external comfort to run this command.");
                    return false;
                }

                if (externalComfort.SimulationResult.ShadeMaterial == null)
                {
                    BH.Engine.Base.Compute.RecordError($"Please provide a valid shade material to the simulation result in external comfort to run this command.");
                    return false;
                }

                if (externalComfort.Typology == null)
                {
                    BH.Engine.Base.Compute.RecordError($"Please provide a valid Typology to the external comfort to run this command.");
                    return false;
                }
            }

            return true;
        }
    }
}
