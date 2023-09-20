set mypath=%cd%
cd C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\src
call C:\ProgramData\BHoM\Extensions\PythonEnvironments\LadybugTools_Toolkit\Scripts\activate.bat
sphinx-apidoc -f -e -d 4 -o ./docs ./ladybugtools_toolkit
sphinx-build -b html ./docs ./docs/_build/docs
cd %mypath%
call C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\src\docs\_build\docs\index.html
cmd /k