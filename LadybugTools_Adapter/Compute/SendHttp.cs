/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */


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
            HttpResponseMessage message = await httpClient.PostAsync("", content).ConfigureAwait(false);

            if (message.IsSuccessStatusCode)
                return (Encoding.UTF8.GetString(await message.Content.ReadAsByteArrayAsync().ConfigureAwait(false)), true);

            return ("", message.IsSuccessStatusCode);
        }
    }
}
