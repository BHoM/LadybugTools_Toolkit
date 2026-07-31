"""Method to wrap creation of diurnal plots"""
# pylint: disable=C0415,E0401,W0703
import os
import matplotlib.pyplot as plt
from ladybug.epw import EPW
from ladybugtools_toolkit.bhom.wrapped.metadata.collection import collection_metadata
from ladybugtools_toolkit.bhom.wrapped.metadata.plot_information import PlotInformation
from ladybugtools_toolkit.plot.utilities import figure_to_base64
from ladybugtools_toolkit.plot.facades.condensation_risk.heatmap import facade_condensation_risk_chart_table
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("plot/facade_condensation_risk_chart")
def facade_condensation_risk_chart(epw_file: str, thresholds: list[float], save_path: str = None, **kwargs) -> PlotInformation:
    style = os.environ.get("BHOM_style_context", "python_toolkit.bhom")

    epw = EPW(epw_file)
    hcc = epw.dry_bulb_temperature

    fig = facade_condensation_risk_chart_table(epw_file, thresholds, style_context=style)

    pi = PlotInformation(other_data = collection_metadata(hcc))
    image: str = ""

    if save_path == None or save_path == "":
        base64 = figure_to_base64(fig,html=False)
        image = base64
    else:
        fig.savefig(save_path, dpi=300, transparent=True)
        image = save_path
    
    plt.close(fig)
    pi.image = image

    return pi
