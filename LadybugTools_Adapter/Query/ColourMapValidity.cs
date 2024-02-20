using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using BH.oM.LadybugTools;

namespace BH.Adapter.LadybugTools
{
    public static partial class Query
    {
        public static bool ColourMapValidity(this string toValidate)
        {
            ColourMap colourMap = toValidate.ToColourMap();
            if (colourMap == ColourMap.Undefined)
            {
                BH.Engine.Base.Compute.RecordWarning($"The input colourmap: {toValidate}, could not be converted into a known colour map. If matplotlib cannot find a colourmap with this name, it will default to 'YlGnBl'.");
                return false;
            }
            return true;
        }
    }
}
