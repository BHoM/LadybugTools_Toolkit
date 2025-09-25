using BH.oM.Adapter;
using System.ComponentModel;

namespace BH.oM.LadybugTools
{
    public class EPWToCSVCommand: IExecuteCommand
    {
        [Description("The epw file to convert to a csv file.")]
        public virtual FileSettings EPWFile { get; set; } = new FileSettings();

        [Description("The directory to place the output csv file.")]
        public virtual string OutputDirectory { get; set; } = "";

        [Description("Whether to include additional calculated values (e.g. sun positions) in the output file.")]
        public virtual bool IncludeAdditionalCalculated { get; set; } = false;
    }
}