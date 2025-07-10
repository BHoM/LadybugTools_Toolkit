
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Compute
    {
        public static (string, bool) RunLBTClientSocket(List<string> args, string host = "127.0.0.1", int port = 5999)
        {
            IPAddress iPAddress = IPAddress.Parse(host);
            IPEndPoint endPoint = new IPEndPoint(iPAddress, port);

            string argString = "[\"" + args.Aggregate((a, b) => a + "\", \"" + b) + "\"]";

            using (Socket client = new Socket(iPAddress.AddressFamily, SocketType.Stream, ProtocolType.Tcp))
            {
                try
                {
                    client.Connect(endPoint);
                    byte[] send = Encoding.UTF8.GetBytes(argString);
                    client.Send(send);

                    //receive everything...
                    byte[] buffer = new byte[1024];
                    int byteCount = client.Receive(buffer);
                    string response = "";

                    while (byteCount > 0)
                    {
                        response += Encoding.UTF8.GetString(buffer, 0, byteCount);
                        byteCount = client.Receive(buffer);
                    }

                    return (response, true);
                }
                catch (SocketException se)
                {
                    BH.Engine.Base.Compute.RecordNote(se, "There was no socket to connect to.");
                    return (se.Message, false);
                }
                catch (Exception e)
                {
                    BH.Engine.Base.Compute.RecordError(e, "An error occurred while trying to connect to the socket.");
                    return (e.Message, false);
                }
            }

        }
    }
}
