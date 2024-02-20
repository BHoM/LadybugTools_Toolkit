using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace BH.Adapter.LadybugTools
{
    public static partial class Query
    {
        public static bool ColourMapValidity(this string toValidate)
        {
            bool valid = false;
            foreach (var item in Enum.GetValues(typeof(oM.LadybugTools.ColourMap)))
            {
                if (toValidate.Equals(item.ToString()))
                    valid = true;
            }
            if (!valid)
            {
                BH.Engine.Base.Compute.RecordWarning("The colour map input is not in the standard list of colour maps. If matplotlib cannot find a colourmap with this name, it will be overridden with 'YlGnBu'.");
            }
            return valid;
        }
    }
}
