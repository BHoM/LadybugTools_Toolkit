# Install LadybugTools_Toolkit Python Environment

# Author: Tristan Gerrish
# Date: 2024-08-21
# Description: This is a batch script to install the LadybugTools_Toolkit outside of BHoM ... if you've got the code on your machine already!

# clear the terminal
Clear-Host

# Set the title of the command prompt window
Write-Host ">>> Installing LadybugTools_Toolkit ..." -ForegroundColor green

# Set constants to use throughout the script
$name = "LadybugTools_Toolkit"
$python_environments_dir = "C:\ProgramData\BHoM\Extensions\PythonEnvironments"
$root_python_exe = "C:\ProgramData\BHoM\Extensions\PythonEnvironments\Python_Toolkit\python.exe"
$lb_python_exe = "C:\Program Files\ladybug_tools\python\python.exe"
$pollination_uninstall_exe = "C:\Program Files\pollination\uninstall.exe"

$target_python_version = "3.10.10"
$target_python_url = "https://www.python.org/ftp/python/3.10.10/python-3.10.10-embed-amd64.zip"

$lbt_env_dir = "C:\ProgramData\BHoM\Extensions\PythonEnvironments\$name"
$lbt_tk_code_dir = "C:\ProgramData\BHoM\Extensions\PythonCode\$name"
$lbttk_python_exe = "$lbt_env_dir\Scripts\python.exe"

# Change to the target directory
cd $python_environments_dir

# if lbt_env_dir exists, then ask the user if they want to overwrite it
if (Test-Path $lbt_env_dir) {
    Write-Host ">>> $name Python environment already exists" -ForegroundColor green
    $overwrite = Read-Host "Do you want to overwrite it? (y/n)"
    if ($overwrite -eq "y") {
        Write-Host ">>> Removing existing $name Python environment" -ForegroundColor green
        Remove-Item -Recurse -Force $lbt_env_dir
    } else {
        Write-Host ">>> Exiting ..." -ForegroundColor green
        exit
    }
}

# check if the BHoM LadybugTools_Toolkit code exists
if (-Not (Test-Path $lbt_tk_code_dir)) {
    Write-Host ">>> $lbt_tk_code_dir not found" -ForegroundColor green
    exit
}

# check if the python executable exists
if (-Not (Test-Path $lb_python_exe)) {
    Write-Host ">>> Ladybug Python executable not found at $lb_python_exe" -ForegroundColor green
    exit
}

# check if the Ladybug Tools Python executable exists
if (-Not (Test-Path $root_python_exe)) {
    Write-Host ">>> BHoM Python executable not found at $root_python_exe" -ForegroundColor green
    exit
}

# check if the Pollination uninstall executable exists, and if so get the version
if (Test-Path $pollination_uninstall_exe) {
    $pollination_version = (Get-Command $pollination_uninstall_exe).FileVersionInfo.ProductVersion
    Write-Host ">>> Pollination version $pollination_version found" -ForegroundColor green
} else {
    Write-Host ">>> Pollination installation not found" -ForegroundColor green
    exit
}

# Get the version of Python from the lb_python_exe
$lb_python_version = & $lb_python_exe --version 2>&1 | Select-String -Pattern "Python (\d+\.\d+\.\d+)"
$lb_python_version = $lb_python_version -replace "Python ", ""
Write-Host ">>> Ladybug Tools Python version: $lb_python_version" -ForegroundColor green

# check that target_python_version is the same as the version of Python in the Ladybug Tools Python executable
if ($lb_python_version -ne $target_python_version) {
    Write-Host ">>> The version of Python in the Ladybug Tools Python executable ($lb_python_version) is not $target_python_version" -ForegroundColor green
    exit
}

# get the filename of the Python executable, and the directory it will be extracted into
$zip_file = Split-Path -Leaf $target_python_url

# get the zip file name without the extension
$src_python_dir = $zip_file -replace ".zip", ""

# check if the zip file already exists
if (-Not (Test-Path $zip_file)) {
    Write-Host ">>> Downloading $target_python_url to $zip_file" -ForegroundColor green
    Invoke-WebRequest -Uri $target_python_url -OutFile $zip_file
}

# extract the zip file
Write-Host ">>> Extracting $zip_file" -ForegroundColor green
Expand-Archive -Force -Path $zip_file -DestinationPath $src_python_dir

# create the new Python environment
$src_python_exe = "$python_environments_dir\$src_python_dir\python.exe"
Write-Host ">>> Creating new Python environment at $lbt_env_dir" -ForegroundColor green
& $root_python_exe -m virtualenv --python=$src_python_exe $lbt_env_dir

# update pip
Write-Host ">>> Updating pip in the new Python environment" -ForegroundColor green
& $lbttk_python_exe -m pip install --upgrade pip

# install ipykernel, pytest and black to the new Python environment
Write-Host ">>> Installing ipykernel, pytest and black in the new Python environment" -ForegroundColor green
& $lbttk_python_exe -m pip install ipykernel pytest black

# register the environment with the root jupyter ipykernel
Write-Host ">>> Registering the new Python environment with the root Jupyter ipykernel" -ForegroundColor green
& $lbttk_python_exe -m ipykernel install --name={name}

# install the local $lbt_tk_code_dir to the python environment
Write-Host ">>> Installing $name code to the new Python environment" -ForegroundColor green
& $lbttk_python_exe -m pip install -e $lbt_tk_code_dir

# create requirements.txt file from the lb_python_exe
Write-Host ">>> Creating requirements.txt file from the Ladybug Tools Python environment" -ForegroundColor green
& $lb_python_exe -m pip freeze > "$lbt_tk_code_dir\lb_requirements.txt"

# remove version from matplotlib, in an attempt to fix the issue with the version of matplotlib
(Get-Content "$lbt_tk_code_dir\lb_requirements.txt") | ForEach-Object { $_ -replace "matplotlib==3.9.1", "matplotlib" } | Set-Content "$lbt_tk_code_dir\lb_requirements.txt"

# install the requirements.txt file to the new lbttk_python_exe
Write-Host ">>> Installing requirements.txt file to the new Python environment" -ForegroundColor green
& $lbttk_python_exe -m pip install -r "$lbt_tk_code_dir\lb_requirements.txt"

# sign off with a message
Write-Host ">>> Installation complete! Trying running $lbt_tk_code_dir\run_tests.bat to check everything went smoothly :)" -ForegroundColor green
