from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..helpers import angle_from_cardinal, cardinality, rolling_window


class DirectionBins:
    """An object containing directional binning data, used mainly for Wind data collections. These bins assume North is at 0-degrees.

    Args:
        directions (int, optional):
            The number of direction bins that should be created. Defaults to 8.
        centered (bool, optional):
            Whether the data should be centered about North - True, or starting from North - False. Defaults to True.

    Returns:
        DirectionBins:
            The resulting object.
    """

    def __init__(
        self,
        directions: int = 8,
        centered: bool = True,
    ) -> DirectionBins:
        self.directions = directions
        self.centered = centered
        self.bins = self.direction_bin_edges(self.directions, self.centered)

    def __len__(self) -> int:
        return self.directions

    @property
    def lows(self) -> List[float]:
        """The "left-most" edge of the direction bins."""
        return self.bins.T[0]

    @property
    def highs(self) -> List[float]:
        """The "right-most" edge of the direction bins."""
        return self.bins.T[1]

    @property
    def midpoints(self) -> List[float]:
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
    def cardinal_directions(self) -> List[str]:
        """The direction bins as cardinal directions."""
        return [cardinality(i, directions=32) for i in self.midpoints]

    @staticmethod
    def direction_bin_edges(directions: int, centered: bool) -> List[List[float]]:
        """Create a list of start/end points for wind directions, each bin increasing from the previous one.
            This method assumes that North is at 0 degrees.
            If the direction bins cross North then the bins returned are all increasing, with the one crossing north
            split into two (and placed at either end of the list of bins).

        Args:
            directions (int, optional):
                The number of directions to bin wind data into.
            centered (bool, optional):
                Whether the data should be centered about North - True, or starting from North - False. Defaults to True.

        Returns:
            List[List[float]]:
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

    def bin_data(
        self, direction_data: List[float], other_data: List[float] = None
    ) -> Dict[Tuple[float], List[float]]:
        """Bin a set of input data, including combination of split bins if present.

        Args:
            direction_data (List[float]):
                A list of wind directions.
            other_data (List[float], optional):
                A list of other data to bin by direction. If None, then direction_data will be used.

        Returns:
            Dict[Tuple[float], List[float]]:
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
                **{k: v for k, v in list(d.items())[1:-1]},
            }
        return d

    def prevailing(
        self, direction_data: List[float], n: int, as_angle: bool = False
    ) -> List[str]:
        """Given a list of wind directions, return the n-prevailing directions.

        Args:
            direction_data (List[float]):
                A set of wind directions.
            n (int):
                The number of prevailing directions to return.
            as_angle (bool, optional):
                Return the previaling directions as angles. Defaults to False.

        Returns:
            List[str]: _description_
        """
        d = self.bin_data(direction_data)
        dd = {}
        for k, v in d.items():
            dd[k] = len(v)
        s = pd.DataFrame.from_dict(dd, orient="index").squeeze()
        s.index = self.cardinal_directions
        s = s.sort_values(ascending=False)

        cardinal_directions = s.index[:n]
        if as_angle:
            return [angle_from_cardinal(i) for i in cardinal_directions]
        return cardinal_directions
