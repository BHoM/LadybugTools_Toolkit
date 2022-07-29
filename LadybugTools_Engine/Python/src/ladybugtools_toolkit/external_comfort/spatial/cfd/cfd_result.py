from __future__ import annotations

from pathlib import Path


class CFDResult:
    """An object containing results from a CFD simulation.
    # These results should be extracted from a number of CFD simulations at the same point locations
    # as in the SpatialComfort simulation, for upwards of 8 (cardinal) directions. Results should be saved in the
    """

    def __init__(self) -> CFDResult:
        self.i = None
    
    @classmethod
    def from_directory(cls, cfd_directory: Path) -> CFDResult:
        return None
    
    def wind
