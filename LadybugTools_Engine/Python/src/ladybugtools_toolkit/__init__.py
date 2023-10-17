"""Base module for the ladybugtools_toolkit package."""
# pylint: disable=E0401
import getpass
import os
from pathlib import Path

# pylint: disable=E0401

import matplotlib.pyplot as plt

# override "HOME" in case IT has set this to something other than default
os.environ["HOME"] = (Path("C:/Users/") / getpass.getuser()).as_posix()

# set plotting style for modules within this toolkit
plt.style.use(Path(__file__).parent / "bhom" / "bhom.mplstyle")
