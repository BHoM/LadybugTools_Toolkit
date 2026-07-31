"""Method to wrap creation of sunpath plots"""
# pylint: disable=C0415,E0401,W0703
import os
import traceback
from ladybugtools_toolkit.plot._sunpath import sunpath as spath
from ladybug.epw import EPW, AnalysisPeriod
from ladybug.sunpath import Sunpath
from ladybugtools_toolkit.bhom.wrapped.metadata.sunpath_metadata import sunpath_metadata
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/sunpath", argument_types = { "analysis_period": AnalysisPeriod }, decoder_cls=LBTBHoMJSONDecoder)
def sunpath(epw_file: str, analysis_period: AnalysisPeriod, size: int, save_path: str = None, **kwargs) -> PlotInformation:
    try:
        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")
        epw = EPW(epw_file)

        with plt.style.context(style):
            fig, ax = plt.subplots()
            spath(location=epw.location, analysis_period=analysis_period, sun_size=size, ax=ax, style_context=style)
            plt.tight_layout()

        pi = PlotInformation(other_data = sunpath_metadata(Sunpath.from_location(epw.location)))
        image: str = ""

        if save_path is None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            image = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            image = save_path
        
        pi.image = image
        plt.close(fig)
        return pi

    except Exception:
        CONSOLE_LOGGER.error("Sunpath could not be created.", exc_info=1)
        return traceback.format_exc()
