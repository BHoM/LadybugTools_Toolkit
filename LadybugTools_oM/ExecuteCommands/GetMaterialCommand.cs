using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("Command that when executed with the LadybugTools Adapter, returns a list of materials from the Python Materials list.")]
    public class GetMaterialCommand : ILadybugCommand, IObject
    {
        [Description("Text to filter the resultant list by. Filter applies to the Material Name. Leave blank to return all Materials.")]
        public virtual string Filter { get; set; } = "";
    }
}
