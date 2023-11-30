using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that when executed with the LadybugTools Adapter, returns a list of Typology objects from the Python predefined Typologies list.")]
    public class GetTypologyCommand : ILadybugCommand, IObject
    {
        [Description("Text to filter the resultant list by. Filter applies to the Typology Name. Leave blank to return all Typologies.")]
        public static string Filter { get; set; } = "";
    }
}
