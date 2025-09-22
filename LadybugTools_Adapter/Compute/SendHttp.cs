
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Net;
using System.Net.Http;
using System.Net.Security;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace BH.Adapter.LadybugTools
{
    public static partial class Compute
    {
        public static async Task<(string, bool)> SendHttp(this HttpClient httpClient, List<string> args, string json = "")
        {
            string argString = "[\"" + args.Aggregate((a, b) => a + "\", \"" + b) + "\"]";
            string start = $"{argString.Length};{json.Length};";

            StringContent content = new StringContent(start + argString + json);
            HttpResponseMessage message = await httpClient.PostAsync("", content);

            if (message.IsSuccessStatusCode)
                return (Encoding.UTF8.GetString(await message.Content.ReadAsByteArrayAsync()), true);

            return ("", message.IsSuccessStatusCode);
        }
    }
}
