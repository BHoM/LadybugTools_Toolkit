using BH.oM.LadybugTools;
using System;
using BH.Engine.Serialiser;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using BH.oM.Adapter;
using System.IO;
using BH.Engine.Adapter;
using BH.oM.Base;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ILadybugTools ToBHoM(this FileSettings jsonFile)
        {
            string json = File.ReadAllText(jsonFile.GetFullFileName());
            ILBTSerialisable LBTObject = Engine.Serialiser.Convert.FromJson(json) as ILBTSerialisable;

            return Deserialise(LBTObject as dynamic);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.AnalysisPeriod LBTObject)
        {
            return AnalysisPeriod(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.DataType LBTObject)
        {
            return DataType(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.EnergyMaterial LBTObject)
        {
            return EnergyMaterial(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.EnergyMaterialVegetation LBTObject)
        {
            return EnergyMaterialVegetation(LBTObject);
        }

        private static ILadybugTools Deserialise
    }
}