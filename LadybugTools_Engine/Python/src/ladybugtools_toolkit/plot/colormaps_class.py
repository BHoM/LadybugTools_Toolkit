from matplotlib.colors import BoundaryNorm, ListedColormap
from typing import List

class UTCIColorScheme:
    """A UTCI color scheme class, including all related information for UTCI.

    Args:
        name (str): The name of the UTCI color scheme.
        levels (List[float]): A list of values for UTCI boundaries in order. Defaults to None. The length of levels should be bigger than 0.
        labels ((List[str]): A list of names for UTCI categories in order. Defaults to None. The length of labels should be one more than the length of levels.
        colors (List[str]): A list of hex color code in order. Defaults to None. The length of color should be one more than the length of levels.

    Returns:
        UTCIColorScheme: A UTCI color scheme.
    """

    def __init__(
        self,
        name: str,
        levels: List[float] = None,
        labels: List[str] = None,
        colors: List[str] = None,
    ):

        if levels is None:
            raise ValueError("The levels cannot be empty.")

        if labels is None:
            raise ValueError("The levels cannot be empty.")

        if colors is None:
            raise ValueError("The levels cannot be empty.")
        
        if len(levels) == 0:
            raise ValueError("The length of levels should be bigger than 0.")

        if len(labels) != (len(levels) + 1):
            raise ValueError("The length of labels should be one more than the length of levels.")
            
        if len(colors) != (len(levels) + 1):
            raise ValueError("The length of color should be one more than the length of levels.")

        self.name = name
        self.UTCI_LEVELS = levels
        self.UTCI_LEVELS_IP = [ x * 1.8 + 32 for x in levels]
        self.UTCI_LABELS = labels
        self.UTCI_COLORMAP = ListedColormap(colors[1:-1])
        self.UTCI_COLORMAP.set_under(colors[0])
        self.UTCI_COLORMAP.set_over(colors[-1])
        self.UTCI_BOUNDARYNORM = BoundaryNorm(self.UTCI_LEVELS, self.UTCI_COLORMAP.N)
        self.UTCI_BOUNDARYNORM_IP = BoundaryNorm(self.UTCI_LEVELS_IP, self.UTCI_COLORMAP.N)


    def __repr__(self):
        return f"{self.__class__.__name__} - {self.name}"