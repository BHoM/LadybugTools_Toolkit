using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public class LadybugConfig : ActionConfig
    {
        public virtual FileSettings JsonFile { get; set; } = null;
    }
}
