"""Method to wrap for creating wind roses from epw files."""
# pylint: disable=C0415,E0401,W0703
import argparse
import os
import sys
import traceback
from pathlib import Path
import matplotlib
from ladybug.epw import EPW, AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.wind import Wind
from ladybugtools_toolkit.bhom.wrapped.metadata.wind_metadata import wind_metadata
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import matplotlib.pyplot as plt
from pathlib import Path
import json
from ...logger import CONSOLE_LOGGER
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/windrose", argument_types = { "analysis_period": AnalysisPeriod }, decoder_cls=LBTBHoMJSONDecoder)
def windrose(epw_file: str, analysis_period: AnalysisPeriod, colour_map: str, bins: int, save_path: str = None, **kwargs) -> PlotInformation:
    """Method to wrap for creating wind roses from epw files."""
    try:
        locator = kwargs.pop("epw_locator", None)
        if locator is not None:
            epw_file = locator(epw_file)

        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")

        if colour_map not in plt.colormaps():
            colour_map = "YlGnBu"

        w_epw = Wind.from_epw(epw_file)
        wind_filtered = w_epw.filter_by_analysis_period(analysis_period=analysis_period)

        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})
            wind_filtered.plot_windrose(ax=ax, directions=bins, ylim=(0, 3.6/bins), colors=colour_map, style_context=style)
            plt.tight_layout()
        
        pi = PlotInformation(other_data = wind_metadata(wind_filtered, directions=bins))
        image:str = ""

        if save_path == None or save_path == "":
            image = figure_to_base64(fig,html=False)
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            image = save_path
            
        pi.image = image
        plt.close(fig)
        return pi
    except Exception:
        CONSOLE_LOGGER.error("Windrose could not be created.", exc_info=1)
        return traceback.format_exc()
