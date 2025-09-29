using BH.oM.Adapter;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class HBJSONToGEMCommand: IExecuteCommand
    {
        [Description("The HBJSON file to convert to a GEM file.")]
        public virtual FileSettings HBJSONFile { get; set; } = new FileSettings();

        [Description("The directory to place the output GEM file.")]
        public virtual string OutputDirectory { get; set; } = "";
    }
}