using BH.oM.Reflection.Attributes;

using System.ComponentModel;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        [Description("Get the name of the current toolkit.")]
        [Output("name", "The name of the current toolkit.")]
        public static string ToolkitName()
        {
            return "LadybugTools_Toolkit";
        }
    }
}
