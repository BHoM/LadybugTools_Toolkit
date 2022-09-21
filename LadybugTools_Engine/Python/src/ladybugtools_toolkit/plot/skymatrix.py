﻿import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.epw import EPW
from ladybug.viewsphere import ViewSphere
from ladybug.wea import AnalysisPeriod, Wea
from ladybugtools_toolkit.external_comfort.simulate import hbr_folders
from ladybugtools_toolkit.ladybug_extension.analysis_period.describe import (
    describe as describe_ap,
)
from ladybugtools_toolkit.ladybug_extension.location.to_string import (
    to_string as describe_loc,
)
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap
from matplotlib.figure import Figure


from ladybugtools_toolkit import analytics


@analytics
def skymatrix(
    epw: EPW,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    density: int = 1,
    cmap: Union[Colormap, str] = "viridis",
    show_title: bool = True,
    show_colorbar: bool = True,
) -> Figure:
    """Create a sky matrix image.

    Args:
        epw (EPW):
            A EPW object.
        analysis_period (AnalysisPeriod, optional):
            An AnalysisPeriod. Defaults to AnalysisPeriod().
        density (int, optional):
            Sky matrix density. Defaults to 1.
        cmap (Union[Colormap, str], optional):
            A colormap. Defaults to "viridis".
        show_title (bool, optional):
            Show the title. Defaults to True.
        show_colorbar (bool, optional):
            Show the colorbar. Defaults to True.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    # create wea
    wea = Wea.from_epw_file(
        epw.file_path, analysis_period.timestep
    ).filter_by_analysis_period(analysis_period)
    wea_duration = len(wea) / wea.timestep
    wea_folder = Path(tempfile.gettempdir())
    metd = epw.direct_normal_radiation.header.metadata
    wea_path = wea_folder / "skymatrix.wea"
    wea_file = wea.write(wea_path.as_posix())

    # run gendaymtx
    gendaymtx_exe = (Path(hbr_folders.radbin_path) / "gendaymtx.exe").as_posix()
    cmds = [gendaymtx_exe, "-m", str(density), "-d", "-O1", "-A", wea_file]
    process = subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True)
    stdout = process.communicate()
    dir_data_str = stdout[0].decode("ascii")
    cmds = [gendaymtx_exe, "-m", str(density), "-s", "-O1", "-A", wea_file]
    process = subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True)
    stdout = process.communicate()
    diff_data_str = stdout[0].decode("ascii")

    def _broadband_rad(data_str: str) -> List[float]:
        _ = data_str.split("\r\n")[:8]
        data = np.array(
            [[float(j) for j in i.split()] for i in data_str.split("\r\n")[8:]][1:-1]
        )
        patch_values = (np.array([0.265074126, 0.670114631, 0.064811243]) * data).sum(
            axis=1
        )
        patch_steradians = np.array(ViewSphere().dome_patch_weights(density))
        broadband_radiation = patch_values * patch_steradians * wea_duration / 1000
        return broadband_radiation

    dir_vals = _broadband_rad(dir_data_str)
    diff_vals = _broadband_rad(diff_data_str)

    # create patches to plot
    patches = []
    for face in ViewSphere().dome_patches(density)[0].face_vertices:
        patches.append(mpatches.Polygon(np.array([i.to_array() for i in face])[:, :2]))
    p = PatchCollection(patches, alpha=1, cmap=cmap)

    p.set_array(dir_vals + diff_vals)  # SET DIR/DIFF/TOTAL VALUES HERE

    # plot!
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.add_collection(p)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    if show_colorbar:
        cbar = fig.colorbar(p, ax=ax)
        cbar.outline.set_visible(False)
        cbar.set_label("Cumulative irradiance (W/m$^{2}$)")
    ax.set_aspect("equal")
    ax.axis("off")

    if show_title:
        ax.set_title(
            f"{describe_loc(epw.location)}\n{describe_ap(analysis_period)}",
            ha="left",
            x=0,
        )

    plt.tight_layout()

    return fig
