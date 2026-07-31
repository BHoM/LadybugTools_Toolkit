"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import os
import traceback
from ladybug.epw import EPW
from ladybugtools_toolkit.plot.compare import compare_epw_key_line, compare_epw_key_hist
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from typing import List
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/epw_comparison")
def epw_comparison(epw_file: str, epw_list: List[str], data_type_key: str, line:bool, save_path:str = None) -> PlotInformation:
    """Create a timeseries plot with a line for each epw file for the specified data key and return it in a format readable by the LadybugToolsAdapter."""
    try:
        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")
        epws = [EPW(epw_file)]
        epws.extend([EPW(f) for f in epw_list])

        with plt.style.context(style):
            fig, ax = plt.subplots()

            if line:
                compare_epw_key_line(epws, key=data_type_key.lower().strip().replace(" ", "_"), style_context=style, ax=ax)
            else:
                compare_epw_key_hist(epws, key=data_type_key.lower().strip().replace(" ", "_"), style_context=style, ax=ax)

        pi = PlotInformation() #Unsure of how to create representative collection metadata for a comparison plot type that doesn't simply list every epw file compared
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
        CONSOLE_LOGGER.error("Timeseries comparison could not be created.", exc_info=1)
        return traceback.format_exc()
