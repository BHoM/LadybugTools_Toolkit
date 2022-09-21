import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


from ladybugtools_toolkit import analytics


@analytics
def utci_day_comfort_metrics(
    utci: pd.Series,
    dbt: pd.Series,
    mrt: pd.Series,
    rh: pd.Series,
    ws: pd.Series,
    month: int = 6,
    day: int = 21,
    title: str = None,
) -> Figure:
    """Plot a single days UTCI with composite DBT, RH, MRT and WS components shown also.

    Args:
        utci (pd.Series):
            An annual time-indexed series containing UTCI values.
        dbt (pd.Series):
            An annual time-indexed series containing DBT values.
        mrt (pd.Series):
            An annual time-indexed series containing MRT values.
        rh (pd.Series):
            An annual time-indexed series containing RH values.
        ws (pd.Series):
            An annual time-indexed series containing WS values.
        month (int, optional):
            The month to plot. Default is 6.
        day (int, optional):
             The day to plot. Default is 21.
        title (str, optional):
            A title to be added to the resulting figure. Default is None.

    Returns:
        Figure:
            A matplotlib Figure object.
    """

    if any([all(utci.index != i.index) for i in [dbt, mrt, rh, ws]]):
        raise ValueError("All series must have the same index")

    try:
        dt = f"{utci.index.year[0]}-{month}-{day}"
        date = utci.loc[dt].index[0]
    except KeyError as e:
        raise e

    fig, ax = plt.subplots(figsize=(10, 4))

    axes = []
    for i in range(5):
        if i == 0:
            axes.append(ax)
        else:
            temp_ax = ax.twinx()
            rspine = temp_ax.spines["right"]
            rspine.set_position(("axes", 0.88 + (0.12 * i)))
            temp_ax.set_frame_on(True)
            temp_ax.patch.set_visible(False)
            axes.append(temp_ax)

    (a,) = axes[0].plot(utci.loc[dt], c="black", label="UTCI")
    axes[0].set_ylabel("UTCI")
    (b,) = axes[1].plot(dbt.loc[dt], c="red", alpha=0.75, label="DBT", ls="--")
    axes[1].set_ylabel("DBT")
    (c,) = axes[2].plot(mrt.loc[dt], c="orange", alpha=0.75, label="MRT", ls="--")
    axes[2].set_ylabel("MRT")
    (d,) = axes[3].plot(rh.loc[dt], c="blue", alpha=0.75, label="RH", ls="--")
    axes[3].set_ylabel("RH")
    (e,) = axes[4].plot(ws.loc[dt], c="green", alpha=0.75, label="WS", ls="--")
    axes[4].set_ylabel("WS")

    for ax in axes:
        ax.spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        axes[0].spines[spine].set_color("k")

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_xlim(utci.loc[dt].index.min(), utci.loc[dt].index.max())

    lgd = axes[0].legend(
        handles=[a, b, c, d, e],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=[0.5, -0.25],
        frameon=False,
    )
    lgd.get_frame().set_facecolor((1, 1, 1, 0))

    for text in lgd.get_texts():
        plt.setp(text, color="k")

    if title:
        ax.set_title(f"{date:%B %d} - {title}", ha="left", va="bottom", x=0)

    plt.tight_layout()

    return fig
