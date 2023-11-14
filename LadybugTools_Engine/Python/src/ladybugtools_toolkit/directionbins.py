"""Methods for working with time-indexed wind data."""
# pylint: disable=E0401
from dataclasses import dataclass
import json
from pathlib import Path

# pylint: enable=E0401
import numpy as np
import pandas as pd

from .bhom.analytics import bhom_analytics


from .helpers import (
    angle_from_cardinal,
    cardinality,
    rolling_window,
)


@dataclass(init=True, repr=True, eq=True)
class DirectionBins:
    """An object containing directional binning data, used mainly for Wind data
    collections. These bins assume North is at 0-degrees.

    Args:
        directions (int, optional):
            The number of direction bins that should be created. Defaults to 8.
        centered (bool, optional):
            Whether the data should be centered about North - True, or starting from
            North - False. Defaults to True.

    Returns:
        DirectionBins:
            The resulting object.
    """

    directions: int = 8
    centered: bool = True

    def __len__(self) -> int:
        return self.directions

    def __str__(self) -> str:
        return f"DirectionBins_{self.directions}_{self.centered}"

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""
        return {
            "_t": "BH.oM.LadybugTools.DirectionBins",
            "directions": self.directions,
            "centered": self.centered,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DirectionBins":
        """Create a DirectionBins object from a dictionary."""
        return cls(
            directions=data["directions"],
            centered=data["centered"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "DirectionBins":
        """Create this object from a JSON string."""

        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")

        with open(Path(path), "w") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_file(cls, path: Path) -> "DirectionBins":
        """Create this object from a JSON file."""
        with open(Path(path), "r") as fp:
            return cls.from_json(fp.read())

    @property
    def bins(self) -> list[list[float]]:
        """The direction bins as a list of lists."""
        return self.direction_bin_edges(self.directions, self.centered)

    @property
    def lows(self) -> list[float]:
        """The "left-most" edge of the direction bins."""
        return self.bins.T[0]

    @property
    def highs(self) -> list[float]:
        """The "right-most" edge of the direction bins."""
        return self.bins.T[1]

    @property
    def midpoints(self) -> list[float]:
        """The mipoints within the direction bins."""
        if self.is_split:
            return np.concatenate([[0], np.mean(self.bins, axis=1)[1:-1]])
        return np.mean(self.bins, axis=1)

    @property
    def bin_width(self) -> float:
        """The width of each direction bin."""
        return 360 / self.directions

    @property
    def is_split(self) -> bool:
        """True if the "north" bin is split into two parts - before and after north."""
        return len(self.bins) != self.directions

    @property
    def cardinal_directions(self) -> list[str]:
        """The direction bins as cardinal directions."""
        return [cardinality(i, directions=32) for i in self.midpoints]

    @staticmethod
    def direction_bin_edges(directions: int, centered: bool) -> list[list[float]]:
        """Create a list of start/end points for wind directions, each bin increasing from the previous one.
            This method assumes that North is at 0 degrees.
            If the direction bins cross North then the bins returned are all increasing, with the one crossing north
            split into two (and placed at either end of the list of bins).

        Args:
            directions (int, optional):
                The number of directions to bin wind data into.
            centered (bool, optional):
                Whether the data should be centered about North - True, or starting from
                North - False. Defaults to True.

        Returns:
            list[list[float]]:
                A set of bin edges.
        """

        if directions < 2:
            raise ValueError("n_directions must be ≥ 2.")

        # get angle for each bin
        bin_angle = 360 / directions

        # get all bin edge angles for 0-360deg
        bin_edges = np.arange(
            0 if not centered else 0 - (bin_angle / 2), 360, bin_angle
        )

        # replace -Ve values with the 360-X version
        bin_edges = np.where(bin_edges < 0, 360 + bin_edges, bin_edges)

        # add north to sequence if required
        if not centered:
            bin_edges = np.concatenate([bin_edges, [360]])

        # create pairs of edges for each bin
        bin_edges = rolling_window(bin_edges, 2)

        # if the start bin crosses "north", then split it into two - either end of the array
        if bin_edges[0][0] > bin_edges[0][-1]:
            bin_edges = np.array(
                [[0, bin_edges[0][1]]]
                + bin_edges[1:].tolist()
                + [[bin_edges[-1][1], 360]]
            )

        return bin_edges

    @property
    def interval_index(self) -> pd.IntervalIndex:
        """The direction bins as a pandasa IntervalIdex."""
        return pd.IntervalIndex.from_arrays(self.lows, self.highs, closed="left")

    @bhom_analytics()
    def bin_data(
        self, direction_data: list[float], other_data: list[float] = None
    ) -> dict[tuple[float], list[float]]:
        """Bin a set of input data, including combination of split bins if present.

        Args:
            direction_data (list[float]):
                A list of wind directions.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then direction_data will be used.

        Returns:
            dict[tuple[float], list[float]]:
                A dictionary indexed by direction bin edges, and containing values.
        """

        if other_data is None:
            other_data = direction_data

        direction_data = np.array(direction_data)  # type: ignore
        other_data = np.array(other_data)  # type: ignore

        if len(direction_data) != len(other_data):
            raise ValueError(
                f"direction_data and other_data must be the same length! ({len(direction_data)} != {len(other_data)})"
            )

        # create bin labels
        bin_labels = list(zip(*[self.lows, self.highs]))  # type: ignore

        # bin data
        temp = pd.DataFrame([direction_data, other_data]).T
        d = {}
        for low, high in bin_labels:
            d[(low, high)] = temp[1][(temp[0] > low) & (temp[0] <= high)].tolist()

        # combine split bins if present
        if self.is_split:
            combined_bin = d[bin_labels[0]] + d[bin_labels[-1]]
            d = {
                **{(bin_labels[-1][0], bin_labels[0][1]): combined_bin},
                **dict(list(d.items())[1:-1]),
            }
        return d

    @bhom_analytics()
    def prevailing(
        self, direction_data: list[float], n: int, as_angle: bool = False
    ) -> list[str]:
        """Given a list of wind directions, return the n-prevailing directions.

        Args:
            direction_data (list[float]):
                A set of wind directions.
            n (int):
                The number of prevailing directions to return.
            as_angle (bool, optional):
                Return the previaling directions as angles. Defaults to False.

        Returns:
            list[str]: _description_
        """
        d = self.bin_data(direction_data)
        dd = {}
        for k, v in d.items():
            dd[k] = len(v)
        s = pd.DataFrame.from_dict(dd, orient="index").squeeze()
        s.index = self.cardinal_directions
        s = s.sort_values(ascending=False)

        cardinal_directions = s.index[:n].tolist()
        if as_angle:
            return [angle_from_cardinal(i) for i in cardinal_directions]
        return cardinal_directions