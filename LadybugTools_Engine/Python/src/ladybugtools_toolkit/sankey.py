import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from ladybug.epw import EPW
from matplotlib.colors import to_hex, to_rgba
from python_toolkit.bhom.logging import CONSOLE_LOGGER
from tqdm import tqdm


def the_helpful_sankey_data_maker_function(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Takes a pandas DataFrame and returns a dictionary of data for a Sankey diagram.

    Args:
        dataframe (pd.DataFrame):
            A pandas DataFrame with columns for the source, target, and value
            of each flow. Plus a few others for formatting of the links between nodes.

    Returns:
        dict[str, Any]:
            A dictionary with the data needed to create a Sankey diagram in Plotly.
    """

    # validation
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame.")

    # ensure columns are present
    required_columns = ["source", "target", "value", "label", "color"]
    if not all(col in dataframe.columns for col in required_columns):
        raise ValueError(f"dataframe must have columns for {required_columns}.")

    # create dict of each node, and input/output flow
    node_flows = {}
    for node in (
        pd.concat([dataframe["source"], dataframe["target"]], axis=0)
        .drop_duplicates()
        .values.tolist()
    ):
        if node not in node_flows:
            node_flows[node] = {"in": 0, "out": 0}
        node_flows[node]["out"] = dataframe["value"][dataframe["source"] == node].sum()
        node_flows[node]["in"] = dataframe["value"][dataframe["target"] == node].sum()

    # raise warning if flows are imbalanced
    for node, v in node_flows.items():
        if v["in"] != v["out"]:
            CONSOLE_LOGGER.warning(
                "%s has imbalanced flows [IN: %s, OUT: %s].", node, v["in"], v["out"]
            )

    # create a lookup for integers to nodes
    node_lookup = {node: i for i, node in enumerate(node_flows.keys())}

    # create link colors
    colors = []
    for color, alpha in dataframe[["color", "alpha"]].values.tolist():
        rgba = to_rgba(color, alpha) * 255
        colors.append(f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]})")

    # generate the dictionary containing the Sankey diagram data
    sankey_data = {
        "node": {
            "pad": 15,
            "thickness": 20,
            "line": {"color": "black", "width": 0.5},
            "label": list(node_flows.keys()),
            "color": ["black" for _ in node_flows.keys()],
        },
        "link": {
            "source": dataframe["source"].map(node_lookup).tolist(),
            "target": dataframe["target"].map(node_lookup).tolist(),
            "value": dataframe["value"].tolist(),
            "label": dataframe["label"].tolist(),
            "color": colors,
        },
    }
    print(sankey_data)
    return sankey_data


if __name__ == "__main__":

    # load CSV into pandas DataFrame
    csv_file = r"C:\Users\tgerrish\Downloads\sankey_example.csv"
    df = pd.read_csv(csv_file)

    # create the Sankey diagram data
    sankey_data = the_helpful_sankey_data_maker_function(df)

    # create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(node=sankey_data["node"], link=sankey_data["link"])])
    # fig = go.Figure(
    #     data=[
    #         go.Sankey(
    #             node={
    #                 "pad": 15,
    #                 "thickness": 20,
    #                 "line": {"color": "black", "width": 0.5},
    #                 "label": [
    #                     "A",
    #                     "B",
    #                     "C",
    #                     "D",
    #                 ],
    #                 "color": ["black", "black", "black", "black"],
    #             },
    #             link={
    #                 "source": [0, 1, 2, 3],  # indices correspond to labels, eg A1, A2, A1, B1, ...
    #                 "target": [2, 2, 3, 4],
    #                 "value": [0.25, 1, 1.25, 1.25],
    #             },
    #         )
    #     ]
    # )

    fig.update_layout(title_text="Example", font_size=30)
    # pio.show(fig)
    fig.show()
