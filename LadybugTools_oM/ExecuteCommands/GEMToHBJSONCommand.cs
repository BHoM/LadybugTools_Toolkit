using BH.oM.Adapter;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class GEMToHBJSONCommand: IExecuteCommand
    {
        [Description("The GEM file to convert to an HBJSON file.")]
        public virtual FileSettings GEMFile { get; set; } = new FileSettings();

        [Description("The directory to place the output HBJSON file.")]
        public virtual string OutputDirectory { get; set; } = "";
    }
}