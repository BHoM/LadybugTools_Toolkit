import sys
import os
import getpass

sys.path.insert(0, r"C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit")

"""
Where this code is run and IT policies modify the "HOME" environment variable, 
this part is essential to make sure tyhat HOME is accessible via the Honeybee/
Queenbee configuration.
"""

os.environ["HOME"] = f"C:\\Users\\{getpass.getuser()}"
