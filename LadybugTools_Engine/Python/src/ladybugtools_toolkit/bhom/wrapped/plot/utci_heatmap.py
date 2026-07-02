"""Method to wrap UTCI plots"""
# pylint: disable=C0415,E0401,W0703
import argparse
import os
import sys
import traceback
import matplotlib
from ladybugtools_toolkit.external_comfort.externalcomfort import ExternalComfort
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.bhom.wrapped.metadata.utci_metadata import utci_metadata
from ladybugtools_toolkit.bhom.from_bhom import LBTBHoMJSONDecoder
from ladybugtools_toolkit.bhom.to_bhom import LBTBHoMJSONEncoder
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.categorical.categories import Categorical, UTCI_DEFAULT_CATEGORIES
import matplotlib.pyplot as plt
import numpy as np
import json
from ...logger import CONSOLE_LOGGER
from ... import bhom_callable

PARSER = argparse.ArgumentParser(
    description=(
        "Given an EPW file path, extract a heatmap"
    )
)
PARSER.add_argument(
    "-in",
    "--input_json",
    help="helptext",
    type=str,
    required=True,
)

@bhom_callable(argument_types = { "external_comfort": ExternalComfort }, encoder_cls=LBTBHoMJSONEncoder, decoder_cls=LBTBHoMJSONDecoder)
def utci_heatmap(external_comfort: ExternalComfort, bin_colours: list[str], save_path: str = "") -> dict:
    try:
        style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")
        custom_bins = UTCI_DEFAULT_CATEGORIES

        if len(bin_colours) == 10:
            custom_bins = Categorical(
                bins=(-np.inf, -40, -27, -13, 0, 9, 26, 32, 38, 46, np.inf),
                colors=(bin_colours),
                name="UTCI")

        with plt.style.context(style):
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            external_comfort.plot_utci_heatmap(utci_categories = custom_bins, ax=ax, style_context=style)
            plt.tight_layout()
    
        utci_collection = external_comfort.universal_thermal_climate_index
        pi = PlotInformation(other_data = utci_metadata(utci_collection))

        image:str = ""

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
        CONSOLE_LOGGER.error("UTCI Heatmap could not be created.", exc_info=1)
        return traceback.format_exc()

if __name__ == "__main__":
    args = PARSER.parse_args()
    matplotlib.use("Agg")
    utci_heatmap(args.input_json, args.save_path)