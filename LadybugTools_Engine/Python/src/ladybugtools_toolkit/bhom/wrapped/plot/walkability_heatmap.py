import os
from pathlib import Path
from typing import Dict
import matplotlib
import traceback
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder
from ladybugtools_toolkit.plot.utilities import figure_to_base64
import json
import matplotlib.pyplot as plt
from ...logger import CONSOLE_LOGGER
from python_toolkit.bhom.decorators import bhom_wrapper
from python_toolkit.bhom.bhom_object import BHoMObject

@bhom_wrapper.bhom_callable("plot/walkability_heatmap", argument_types = { "external_comfort": BHoMObject }, encoder_cls=LBTBHoMJSONEncoder, decoder_cls=LBTBHoMJSONDecoder)
def walkability_heatmap(external_comfort: BHoMObject, save_path: str, **kwargs) -> Dict[str, object]:
    try:
        # This mess is to allow the epw file locator to work before external comfort is calculated
        # (as calculation is done on instantiation, the object needs to remain uninstantiated until the epw file is located)
        epw_file = external_comfort.simulation_result.epw_file

        if isinstance(epw_file, BHoMObject):
            if epw_file._t == "BH.oM.Adapter.FileSettings":
                epw_file = Path(epw_file.directory) / epw_file.file_name

        locator = kwargs.pop("epw_locator", None)
        if locator is not None:
            epw_file = locator(epw_file)

        external_comfort.simulation_result.epw_file = epw_file

        external_comfort = ExternalComfort._from_bhom_object(external_comfort)
        # End mess

        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")

        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            external_comfort.plot_walkability_heatmap(ax=ax, style_context=style)
            plt.tight_layout()
        
        image:str = ""

        utci_collection = external_comfort.universal_thermal_climate_index
        pi = PlotInformation(other_data = utci_metadata(utci_collection))

        if save_path == None or save_path == "":
            base64 = figure_to_base64(fig,html=False)
            image = base64
        else:
            fig.savefig(save_path, dpi=150, transparent=True)
            image = save_path
        
        plt.close(fig)
        pi.image = image
        return_dict = {
            "info": pi,
            "external_comfort": external_comfort
        }
        return return_dict
    
    except Exception:
        CONSOLE_LOGGER.error("Walkability plot could not be created.", exc_info=1)
        return traceback.format_exc()
