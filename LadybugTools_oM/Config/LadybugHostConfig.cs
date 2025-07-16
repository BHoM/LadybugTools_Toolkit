using BH.oM.Adapter;
using BH.oM.Base.Attributes;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace BH.oM.LadybugTools.Config
{
    [Description("Config to determine whether to use a host to run a command, and if so where to send the request.")]
    public class LadybugHostConfig : LadybugConfig
    {
        [DisplayText("Use Existing Host")]
        [Description("When running ladybug execute commands, connect to a known host to run the command instead of spawning a new instance of python on the local machine.")]
        public virtual bool UseExistingHost { get; set; } = false;

        [DisplayText("Host Name")]
        [Description("For advanced users: When running ladybug execute commands, connect to this host (can be an ip address or uri)\nDefaults to connecting to localhost (127.0.0.1).")]
        public virtual string HostName { get; set; } = "127.0.0.1";

        [DisplayText("Host Port")]
        [Description("For advanced users: When running ladybug execute commands, connect to the host on this port number. Defaults to 5999")]
        public virtual int HostPort { get; set; } = 5999;
    }
}
