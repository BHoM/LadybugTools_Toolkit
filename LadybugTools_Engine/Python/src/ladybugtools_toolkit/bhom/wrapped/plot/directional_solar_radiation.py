"""Method to wrap creation of panel orientation plots"""
# pylint: disable=C0415,E0401,W0703
import traceback
from pathlib import Path
import os
from ladybugtools_toolkit.solar import IrradianceType, tilt_orientation_factor, create_radiation_matrix
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder
from ladybug.wea import AnalysisPeriod
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.bhom.wrapped.metadata.solar_radiation_metadata import solar_radiation_metadata
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/directional_solar_radiation", argument_types = { "analysis_period": AnalysisPeriod }, decoder_cls=LBTBHoMJSONDecoder)
def directional_solar_radiation(epw_file: str, directions: int, tilts: int, irradiance_type: str, analysis_period: AnalysisPeriod, cmap: str, title: str = None, save_path:str = None, **kwargs) -> PlotInformation:
    try:
        locator = kwargs.pop("epw_locator", None)
        if locator is not None:
            epw_file = locator(epw_file)

        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")

        if cmap not in plt.colormaps():
            cmap = "YlOrRd"

        if irradiance_type == "Total":
            irradiance_type = IrradianceType.TOTAL
        elif irradiance_type == "Diffuse":
            irradiance_type = IrradianceType.DIFFUSE
        elif irradiance_type == "Direct":
            irradiance_type = IrradianceType.DIRECT
        elif irradiance_type == "Reflected":
            irradiance_type = IrradianceType.REFLECTED
            
        values, dirs, tts = create_radiation_matrix(Path(epw_file), rad_type=irradiance_type, analysis_period=analysis_period, directions=directions, tilts=tilts)

        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(22.8/2, 7.6/2))
            tilt_orientation_factor(Path(epw_file), ax=ax, rad_type=irradiance_type, analysis_period=analysis_period, directions=directions, tilts=tilts, cmap=cmap, style_context=style)
            if not (title == "" or title is None):
                ax.set_title(title)
            plt.tight_layout()

        pi = PlotInformation(other_data = solar_radiation_metadata(values, dirs, tts))
        image: str = ""

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            image = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            image = save_path
        
        plt.close(fig)
        pi.image = image
        return pi
    except Exception:
        CONSOLE_LOGGER.error("Solar Radiation plot could not be created.", exc_info=1)
        return traceback.format_exc()