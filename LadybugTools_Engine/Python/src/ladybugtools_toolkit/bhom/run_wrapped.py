from ladybugtools_toolkit.bhom.server_interface import resolve
import sys

def run_wrapped(args):
    res = resolve(args)
    return res

if __name__ == "__main__":
    print(run_wrapped(sys.argv[1:]))