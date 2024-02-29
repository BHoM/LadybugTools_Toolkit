using BH.oM.Base.Attributes;
using BH.oM.LadybugTools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Convert
    {
        public static ColourMap ToColourMap(this string colourMap)
        {
            if (colourMap == null)
            {
                BH.Engine.Base.Compute.RecordError("Cannot convert null string to a colourmap");
                return ColourMap.Undefined;
            }

            foreach (ColourMap item in Enum.GetValues(typeof(ColourMap)))
            {
                List<string> possibleValues = new List<string>();
                possibleValues.Add(item.ToString().ToLower());
                FieldInfo field = item.GetType().GetField(item.ToString());
                DisplayTextAttribute[] array = field.GetCustomAttributes(typeof(DisplayTextAttribute), inherit: false) as DisplayTextAttribute[];
                if (array != null && array.Length > 0)
                    possibleValues.Add(array.First().Text.ToLower());

                if (possibleValues.Any(x => x == colourMap.ToLower()))
                    return item;
            }
            BH.Engine.Base.Compute.RecordError($"Could not convert the input string: {colourMap} to a colourmap.");
            return ColourMap.Undefined;
        }
    }
}
