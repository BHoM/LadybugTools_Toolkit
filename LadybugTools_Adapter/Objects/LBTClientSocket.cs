
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Linq.Expressions;
using System.Net;
using System.Net.Security;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography.X509Certificates;
using System.Text;

namespace BH.Adapter.LadybugTools
{
    public static partial class Compute
    {
        private static bool ValidateServerCert(object sender,
              X509Certificate certificate,
              X509Chain chain,
              SslPolicyErrors sslPolicyErrors)
        {
            if (sslPolicyErrors == SslPolicyErrors.None)
            {
                return true;
            }

            BH.Engine.Base.Compute.RecordError($"Cert Error: {sslPolicyErrors}");
            return false;
        }

        public static (string, bool) RunLBTClientSocket(List<string> args, string inFile = null, string outFile = null, string host = "127.0.0.1", int port = 5999, bool temp = false)
        {
            string argString = "[\"" + args.Aggregate((a, b) => a + "\", \"" + b) + "\"]";
            byte[] argBytes = Encoding.UTF8.GetBytes(argString);
            string firstMessage = $"{argBytes.Length};";
            byte[] file = null;

            if (inFile != null)
            {
                file = System.IO.File.ReadAllBytes(inFile);
                firstMessage += $"{file.Length};";
            }

            else
                firstMessage += "0;";

            try
            {
                using (TcpClient client = new TcpClient(host, port))
                {

                    using (SslStream sslStream = new SslStream(client.GetStream(), false, new RemoteCertificateValidationCallback(ValidateServerCert), null))
                    {
                        sslStream.AuthenticateAsClient(host);
                        //send message containing file length, then send the file, then send the args
                        byte[] first = Encoding.UTF8.GetBytes(firstMessage);
                        sslStream.Write(first, 0, first.Length);
                        sslStream.Flush();

                        sslStream.Write(argBytes, 0, argBytes.Length);
                        sslStream.Flush();

                        if (file != null)
                        {
                            sslStream.Write(file, 0, file.Length);
                            sslStream.Flush();
                        }

                        //receive everything...
                        byte[] buffer = new byte[1024];
                        int byteCount = sslStream.Read(buffer, 0, buffer.Length);
                        string response = "";

                        while (byteCount > 0)
                        {
                            response += Encoding.UTF8.GetString(buffer, 0, byteCount);
                            byteCount = sslStream.Read(buffer, 0, buffer.Length);
                        }

                        return (response, true);
                    }
                }
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
