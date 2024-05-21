using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace BH.Engine.LadybugTools
{
    public static partial class Query
    {
        public static string PythonCodeDirectory()
        {
            string directory = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "PythonCode");

            if (!Directory.Exists(directory))
                return Engine.Python.Query.DirectoryCode();

            return directory;
        }
    }
}
