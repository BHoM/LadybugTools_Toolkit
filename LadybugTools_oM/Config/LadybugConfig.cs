using BH.oM.Adapter;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools
{
    [Description("The action config for the LadybugTools Adapter.")]
    public class LadybugConfig : ActionConfig
    {
        [Description("File settings for the json file to pull/push to.")]
        public virtual FileSettings JsonFile { get; set; } = null;
    }
}
