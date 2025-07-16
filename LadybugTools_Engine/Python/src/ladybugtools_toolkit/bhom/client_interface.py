import socket
import ssl
import sys
import json
from ladybugtools_toolkit.bhom import HOST, PORT
from typing import List

def run_client(args: List[str], host: str = HOST, port: int = PORT):
    """The "client" socket for interaction between bhom c# and python. This could be used with a socket connection from any other language as well.
    Attempts to connect to the given host on the given port with a socket, then json.dumps the args and sends them to the server to handle the command.
    Reads data from the server until the server closes the connection, then returns the data after concatenating all parts.
    If there is a socket error (i.e. failed connection etc.) then defaults to running the command locally instead and returning the output.

    Args:
        args (list[str]):
            the args to send to the server, or use if there is none.
        host (str):
            The host to connect to, defaults to 127.0.0.1
        port (int):
            The port to connect to, defaults to 5999
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall((json.dumps(args)).encode())
            data_full:str = ""
            data = s.recv(1024)
            while data:
                data_full = data_full + data.decode()
                data = s.recv(1024)
            return data_full
    except:
        #lazy import resolver from server interface to handle if the connection failed or no server existed. If this fails, then let the exception through for bhom/c# to handle
        from ladybugtools_toolkit.bhom.server_interface import resolve
        return resolve(args)

if __name__ == "__main__":
    print(run_client(sys.argv[1:]))