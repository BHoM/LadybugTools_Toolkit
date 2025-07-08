import socket
import sys
import json
from ladybugtools_toolkit.bhom import HOST, PORT

def run_client(args):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall((json.dumps(args)).encode())
            data = s.recv(1024)
            while data:
                print(data.decode())
                data = s.recv(1024)
    except socket.error:
        #lazy import resolver from server interface to handle if the connection failed or no server existed. If this fails, then let the exception through for bhom/c# to handle
        from ladybugtools_toolkit.bhom.server_interface import resolve
        resolve(args)

if __name__ == "__main__":
    run_client(sys.argv[1:])