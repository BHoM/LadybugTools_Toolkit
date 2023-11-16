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

            return IDeserialise(LBTObject);
        }

        /*********************************/
        /* Deserialise methods           */
        /*********************************/

        private static ILadybugTools IDeserialise(ILBTSerialisable LBTObject)
        {
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

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.EPW LBTObject)
        {
            return EPW(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.Header LBTObject)
        {
            return Header(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.HourlyContinuousCollection LBTObject)
        {
            return HourlyContinuousCollection(LBTObject);
        }

        private static ILadybugTools Deserialise(BH.Adapter.LadybugTools.Location LBTObject)
        {
            return Location(LBTObject);
        }
        
        /*********************************/
        /* Fallback Methods              */
        /*********************************/

        private static ILadybugTools Deserialise(ILadybugTools LBTObject)
        {
            BH.Engine.Base.Compute.RecordError($"Objects of type: {LBTObject.GetType()}, can not be deserialised.");
            return null;
        }
    }
}