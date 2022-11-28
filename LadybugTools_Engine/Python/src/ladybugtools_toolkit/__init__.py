import getpass
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# override "HOME" in case IT has set this to something other than default
os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# set toolkit name for any logging
TOOLKIT_NAME = "LadybugTools_Toolkit"

# set plotting style for modules within this toolkit
plt.style.use(Path(__file__).parent / "bhomutil" / "bhom.mplstyle")
