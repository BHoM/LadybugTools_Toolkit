SET PYTHON_EXE=C:\ProgramData\BHoM\Extensions\PythonEnvironments\LadybugTools_Toolkit\Scripts\python.exe
SET TEST_DIR=C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\tests

"%PYTHON_EXE%" -m pytest --cov-report term --cov-report html:cov_html --cov ladybugtools_toolkit -v "%C:\ProgramData\BHoM\Extensions\PythonCode\LadybugTools_Toolkit\tests%"
cmd /k