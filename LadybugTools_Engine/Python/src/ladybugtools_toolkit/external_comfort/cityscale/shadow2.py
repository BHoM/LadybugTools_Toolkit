# #this script is used to create the sun shadow plot - called sample.bat
# g.proj -c wkt="C:/Users/tgerrish/AppData/Local/Temp/processing_FPdqCo/9365604b06b9436cbcdc345148260211/crs.prj"
# r.in.gdal input="C:/Users/tgerrish/Documents/GitHub/BHoM/LadybugTools_Toolkit/.ci/unit_tests/python/files/st5872_DSM_1M.tif" band=1 output="rast_62f558e3a467a2" --overwrite -o
# g.region n=173000.0 s=172000.0 e=359000.0 w=358000.0 res=1.0
# r.sunmask elevation=rast_62f558e3a467a2 altitude=45 azimuth=90 east="False" north="False" -z output=outputcafac34d9e884ccfaafaa5f4b5348f5c --overwrite
# g.region raster=outputcafac34d9e884ccfaafaa5f4b5348f5c
# r.out.gdal -t -m input="outputcafac34d9e884ccfaafaa5f4b5348f5c" output="C:/Users/tgerrish/Documents/GitHub/BHoM/LadybugTools_Toolkit/.ci/unit_tests/python/files/st5872_DSM_1M_SUNSHADOW.tif" format="GTiff" createopt="TFW=YES,COMPRESS=LZW" --overwrite

# # it must be run from this command
# grass78.bat --exec "C:/Users/tgerrish/OneDrive - BuroHappold/Desktop/sample.bat"

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def shadow(asc_file: Path, sun_altitude: float, sun_azimuth: float) -> np.ndarray:

    input_file = Path(asc_file)
    output_file = (
        input_file.parent / f"{input_file.stem}_{sun_altitude}_{sun_azimuth}.tif"
    )
    grass_bat = "C:/Program Files/QGIS 3.20.2/bin/grass78.bat"
    exec_bat = "C:/Users/tgerrish/OneDrive - BuroHappold/Desktop/temp.bat"

    prj_file = input_file.parent / "crs.prj"

    # TODO - add location (lat/long into this process for morfe accuracy)
    bat_commands = [
        f'g.proj -c wkt="{prj_file}"',
        f'r.in.gdal input="{input_file}" band=1 output="rast_62f558e3a467a2" --overwrite -o',
        f"g.region n=173000.0 s=172000.0 e=359000.0 w=358000.0 res=1.0",  # TODO: these might need to be inferred from the source file!
        f'r.sunmask elevation=rast_62f558e3a467a2 altitude={sun_altitude} azimuth={sun_azimuth} east="False" north="False" -z output=outputcafac34d9e884ccfaafaa5f4b5348f5c --overwrite',
        f"g.region raster=outputcafac34d9e884ccfaafaa5f4b5348f5c",
        f'r.out.gdal -t -m input="outputcafac34d9e884ccfaafaa5f4b5348f5c" output="{output_file}" format="GTiff" createopt="TFW=YES,COMPRESS=LZW" --overwrite',
    ]

    with open(exec_bat, "w") as fp:
        fp.write("\n".join(bat_commands))

    command = f'"{grass_bat}" --exec "{exec_bat}"'

    output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    # print(output)

    shadow_matrix = np.array(Image.open(output_file.as_posix()))

    # for sun directly overhead, deal with erroneous cases from teh raycasting method
    shadow_matrix = np.where(shadow_matrix < 0, 255, shadow_matrix)

    return shadow_matrix
