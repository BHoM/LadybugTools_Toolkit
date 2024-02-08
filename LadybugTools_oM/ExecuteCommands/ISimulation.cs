using BH.oM.Adapter;
using BH.oM.Base;
using System;
using System.Collections.Generic;
using System.Text;

namespace BH.oM.LadybugTools
{
    public interface ISimulation : IExecuteCommand, IObject
    {
        FileSettings EPWFile { get; set; }
    }
}
