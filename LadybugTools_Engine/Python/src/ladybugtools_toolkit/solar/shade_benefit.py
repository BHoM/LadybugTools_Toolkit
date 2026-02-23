import concurrent.futures
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import plotly.graph_objects as go
from honeybee.config import folders as hb_folders
from ladybug.epw import EPW, AnalysisPeriod, HourlyContinuousCollection
from ladybug.sunpath import Compass, Sunpath
from ladybug_comfort.collection.utci import UTCI
from ladybug_geometry.geometry3d import Ray3D
from tqdm import tqdm

from ..convert.to_color import to_color
from ..convert.to_plotly import to_plotly
from ..json_encoding import AllPowerfulEncoder
from ..ladybug_geometry_extension.util import (
    Face3D,
    Mesh3D,
    Point3D,
    cull_mesh_faces,
    icosphere,
)
from ..bhom.logger import CONSOLE_LOGGER


def _face_values_to_vertex_values(mesh: Mesh3D, face_values: list[float]) -> list[float]:
    """Average face values to each vertex in a Mesh3D."""
    if not isinstance(mesh, Mesh3D):
        raise ValueError("mesh must be a Mesh3D object")
    if len(mesh.faces) != len(face_values):
        raise ValueError("face_values must have the same length as mesh.faces")
    n_vertices = len(mesh.vertices)
    vertex_sums = np.zeros(n_vertices)
    vertex_counts = np.zeros(n_vertices)

    for face_idx, face in enumerate(mesh.faces):
        for vi in face:
            vertex_sums[vi] += face_values[face_idx]
            vertex_counts[vi] += 1

    # Avoid division by zero
    vertex_counts[vertex_counts == 0] = 1
    vertex_values = vertex_sums / vertex_counts
    return vertex_values.tolist()


def utci_thermal_shade_benefit(
    epw_file: Union[Path, str],
    timestep: int = 1,
    dome_resolution: float = 1,
    temperature_lower: float = 9,
    temperature_upper: float = 26,
) -> tuple[Mesh3D, list[float], list[float], list[float]]:

    epw_file = Path(epw_file)
    epw = EPW(epw_file)

    # get all inputs, and hash
    cfg = {}
    for k, v in locals().items():
        if k in ["epw", "cfg", "temperature_lower", "temperature_upper"]:
            continue
        else:
            cfg[k] = v
    h = hashlib.blake2b(digest_size=20, person=b"utci_shd_ben")
    h.update(json.dumps(cfg, sort_keys=True, cls=AllPowerfulEncoder).encode("utf-8"))
    hash_str = h.hexdigest()

    # check if simulation results already exist, and load them instead!
    results_dir = Path(hb_folders.default_simulation_folder) / ".shade_benefit_cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{hash_str}.npz"

    # create dome geometry to assess for shade effectiveness (and to allow it to be returned at the end)
    dome_radii = 90
    sphere = icosphere(resolution=dome_resolution, radius=dome_radii)  # type: ignore
    mask = [i.z > 0 for i in sphere.face_normals]  # type: ignore
    dome = cull_mesh_faces(sphere, mask)

    if results_file.exists():
        CONSOLE_LOGGER.debug(
            f"Loading cached shade benefit results from {results_file}"
        )
        i_matrix = np.load(file=results_file)["i_matrix"]
    else:

        # get each face of the dome as a Face3D object
        dome_faces: list[Face3D] = []
        for face_vertices in dome.face_vertices:
            fc = Face3D.from_array(
                [
                    face_vertices,
                ]
            )
            dome_faces.append(fc)

        # get sun positions and vectors (towards the analysis pt)
        sp = Sunpath.from_location(location=epw.location)
        datetimes = AnalysisPeriod(timestep=timestep).datetimes
        with concurrent.futures.ThreadPoolExecutor() as executor:
            suns = list(executor.map(sp.calculate_sun_from_date_time, datetimes))

        # for each sun-position, determine which of the patches intersects and store the result in a matrix
        origin = Point3D(0, 0, 1.2)  # type: ignore

        # create list of face/sun combinations to process in parallel
        combinations = list(product(range(len(dome_faces)), range(len(suns))))

        # iterate and calculate intersections per sun/face
        results = []
        pbar = tqdm(
            combinations,
            desc="Calculating shade benefit",
            total=len(combinations),
            bar_format="{percentage:3.0f}% |{bar}| ETC {remaining}",
        )
        for face_idx, sun_idx in pbar:
            if suns[sun_idx].position_3d(radius=dome_radii * 2).z <= 0:
                results.append(False)
                continue
            ray = Ray3D(p=origin, v=suns[sun_idx].sun_vector_reversed)
            if dome_faces[face_idx].intersect_line_ray(ray):
                results.append(True)
            else:
                results.append(False)

        # reshape results into matrix form
        i_matrix = np.array(results).reshape(len(dome_faces), len(suns))

        # save to cache
        np.savez_compressed(file=results_file, i_matrix=i_matrix)

    # run the thermal benefit analysis
    utci_obj = UTCI.from_epw(epw=epw, include_wind=True, include_sun=True)
    temperature: HourlyContinuousCollection = (
        utci_obj.universal_thermal_climate_index.interpolate_to_timestep(timestep)
    )
    temperature_array = np.array(temperature)

    # calculate results (the number of degrees above/below the thresholds for each face at each timestep), then sum to get the net benefit/deficit for that face
    result = []
    for intersections in i_matrix:
        for is_shading, temp in zip(intersections, temperature_array):
            if is_shading and temp < temperature_lower:
                # face is shading, and access to sun would be good, so this is a negative benefit (a deficit) and will be a negative number
                result.append(temp - temperature_lower)
            elif is_shading and temp > temperature_upper:
                # face is shading, and access to sun would be bad, so this is a positive benefit and will be a positive number
                result.append(temp - temperature_upper)
            else:
                result.append(0)
    result = np.array(result).reshape(i_matrix.shape)

    # convert from degree-timestep to degree-day
    result /= (timestep * 24 * 365)

    # calculate the benefit and deficit for each face, then sum to get the net benefit/deficit for that face
    benefit = -np.where(result < 0, result.sum(), 0)
    deficit = np.where(result > 0, result.sum(), 0)
    net = benefit + deficit

    return dome, benefit.tolist(), deficit.tolist(), net.tolist()

def utci_thermal_shade_benefit_render(
    epw_file: Union[Path, str],
    timestep: int = 1,
    dome_resolution: float = 1,
    temperature_lower: float = 9,
    temperature_upper: float = 26,
    shade_threshold: float = 0.25,
) -> go.Figure:
    
    if shade_threshold < 0 or shade_threshold > 1:
        raise ValueError("shade_threshold must be between 0 and 1")
    epw_file = Path(epw_file)
    
    # run calculation
    dome, benefit, deficit, net = utci_thermal_shade_benefit(
        epw_file=epw_file,
        timestep=timestep,
        dome_resolution=dome_resolution,
        temperature_lower=temperature_lower,
        temperature_upper=temperature_upper,
    )

    face_values = np.array(net).sum(axis=1)
    vertex_values = np.array(_face_values_to_vertex_values(dome, face_values))
    vmax = max(abs(vertex_values.min()), abs(vertex_values.max()))

    # create coloured mesh dome
    dome_trace = to_plotly(
        dome,
        intensity=vertex_values,
        cmin=-vmax,
        cmax=vmax,
        colorscale="RdBu_r",
        colorbar = dict(
            title="degC-days", 
            x=0.85,      # horizontal position (0=left, 1=right)
            y=0.5,       # vertical position (0=bottom, 1=top)
            len=0.75,    # length of colorbar),
        ),
        opacity=1,
        showlegend=False,
    )[0]

    # generate hoverinfo for mesh
    x, y, z = np.array(
        [
            dome_trace.x,
            dome_trace.y,
            dome_trace.z,
        ]
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # azimuth, in radians
    phi = np.arccos(z / r)  # inclination, in radians
    theta_deg = np.degrees(theta)  # angle "around"
    phi_deg = np.degrees(phi)  # angle "up"
    altitude = 90 - phi_deg
    azimuth = 360 - ((theta_deg - 90) % 360)
    azimuth = np.where(azimuth == 360, 0, azimuth)  # convert 360 to 0 for cleaner display
    dome_trace.hovertext = [
        f"azimuth: {az:.1f}°<br>altitude: {alt:.1f}°<br>{v:0.1f}degC-days"
        for az, alt, v in zip(azimuth, altitude, vertex_values)
    ]
    dome_trace.hoverinfo = "text"
    
    # generate wireframe for mesh dome
    dome_edges_trace = []
    for fe in dome.face_edges:
        dome_edges_trace.extend(
            to_plotly(
                fe, showlegend=False, line=dict(color="black", width=2), hoverinfo="skip"
            )
        )

    # create mesh traces for where shade is most/least acceptable
    no_shade_threshold = np.percentile(face_values[face_values < 0], shade_threshold * 100)
    no_shade_mask = face_values < no_shade_threshold
    no_shade_mesh = cull_mesh_faces(dome, mask=no_shade_mask).scale(factor=0.9)
    no_shade_mesh = to_plotly(
        no_shade_mesh,
        facecolor=[to_color("red", fmt="plotly")] * len(no_shade_mesh.faces),
        opacity=0.5,
        showlegend=True,
        hoverinfo="skip",
        name="Detrimental shade",
    )[0]

    yes_shade_threshold = np.percentile(
        face_values[face_values > 0], 100 - (shade_threshold * 100)
    )
    yes_shade_mask = face_values > yes_shade_threshold
    yes_shade_mesh = cull_mesh_faces(dome, mask=yes_shade_mask).scale(factor=0.9)
    yes_shade_mesh = to_plotly(
        yes_shade_mesh,
        facecolor=[to_color("green", fmt="plotly")] * len(yes_shade_mesh.faces),
        opacity=0.5,
        showlegend=True,
        hoverinfo="skip",
        name="Beneficial shade"
    )[0]

    # create traces for subpath
    sunpath_traces = to_plotly(
        Sunpath.from_location(location=EPW(epw_file).location),
    )

    # create traces for compass
    compass_traces = to_plotly(
        Compass()
    )

    # generate figure
    fig = go.Figure()
    fig.add_traces([dome_trace] + dome_edges_trace + [no_shade_mesh] + [yes_shade_mesh] + sunpath_traces + compass_traces)

    ti = [epw_file.name, "UTCI thermal shade benefit<br>blue = net deficit, red = net benefit<br>"]
    sort_idx = np.argsort(vertex_values)
    arr_sorted = np.vstack([altitude, azimuth, vertex_values])[:, sort_idx]
    top_n = 5
    ti.append(f"Top {top_n} directions to shade to reduce heat stress:")
    for alt, az in zip(arr_sorted[0][-top_n:], arr_sorted[1][-top_n:]):
        ti.append(f"- azimuth: {az:.1f}°, altitude: {alt:.1f}°")
    ti.append("Avoid shading these directions to reduce cold-stress:")
    for alt, az in zip(arr_sorted[0][:top_n], arr_sorted[1][:top_n]):
        ti.append(f"- azimuth: {az:.1f}°, altitude: {alt:.1f}°")

    fig.update_layout(
        title="<br>".join(ti),
        scene_camera=dict(
            eye=dict(x=0, y=0, z=2.5),  # y is "out of screen", so look from +y
            up=dict(x=0, y=1, z=0),  # z is up
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        legend=dict(
            x=1.05,  # move legend further right
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),
    )

    return fig