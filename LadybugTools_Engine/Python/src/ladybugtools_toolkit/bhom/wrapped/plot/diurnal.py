"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import os
import traceback
from ladybug.epw import EPW
from ladybugtools_toolkit.ladybug_extension.datacollection import collection_to_series
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.ladybug_extension.epw import wet_bulb_temperature
from python_toolkit.plot.diurnal import diurnal as dnal
from ladybug.datacollection import HourlyContinuousCollection
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/epw_diurnal")
def diurnal(epw_file: str, data_type_key: str="Dry Bulb Temperature", colour: str="#000000", title: str=None, period: str="monthly", save_path: str=None, **kwargs) -> PlotInformation:
    try:
        locator = kwargs.pop("epw_locator", None)
        if locator is not None:
            epw_file = locator(epw_file)

        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")
        epw = EPW(epw_file)
        
        if data_type_key == "Wet Bulb Temperature":
            coll = wet_bulb_temperature(epw)
        else:
            coll = HourlyContinuousCollection.from_dict([a for a in epw.to_dict()["data_collections"] if a["header"]["data_type"]["name"] == data_type_key][0])

        with plt.style.context(style):
            fig, ax = plt.subplots()
            dnal(collection_to_series(coll), ax=ax, title=title, period=period, color=colour, style_context=style)
        
        pi = PlotInformation(other_data = collection_metadata(coll))
        image: str = ""

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig, html=False)
            image = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            image = save_path

        plt.close(fig)
        pi.image = image
        return pi

    except Exception:
        CONSOLE_LOGGER.error("Diurnal plot could not be created.", exc_info=1)
        return traceback.format_exc()
