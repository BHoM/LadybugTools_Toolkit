import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\Python_Toolkit\src")

from python_toolkit.bhom.analytics import analytics as analytics

os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()
