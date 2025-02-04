"""Configuration options for the processing of climatic data from a pandas DataFrame"""

from typing import Literal, Optional

from ladybug.datatype.base import DataTypeBase, _DataTypeEnumeration
from ladybug.datatype.temperature import DryBulbTemperature
from matplotlib.colors import Colormap
from pydantic import BaseModel, ConfigDict, Field

from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, CategoricalComfort


class MissingVariableException(Exception):
    """Custom exception raised when a required variable is missing."""


class ClimateVariable(BaseDataType):
    """A class containing configuration options for differnet variable types.
    This extends the already useful Ladybug datatype with additinal features
    such as default color, colormapping and variable name.
    """

    name: str = None

    @classmethod
    def from_ladybug(cls, dadybug_datatype: BaseDataType):
        """Create a ClimateVariable from a Ladybug variable."""
        return cls(
            name=ladybug_variable.name,
            color=ladybug_variable.color,
            cmap=ladybug_variable.cmap,
        )


class SummariseClimateConfig(BaseModel):
    """Configuration options for the processing of climatic data from a pandas DataFrame"""

    # plot configurations
    dpi: Optional[int] = Field(default=300)
    extension: Literal["png", "pdf"] = Field(
        default="png",
    )
    # figsizes
    diurnal_figsize: Optional[tuple[float]] = Field(default=(22.8, 6.2))
    windrose_figsize: Optional[tuple[float]] = Field(default=(8, 8))
    windmatrix_figsize: Optional[tuple[float]] = Field(default=(9.9, 6.7188))
    seasonality_figsize: Optional[tuple[float]] = Field(default=(15.048, 4.752))
    sunriseset_figsize: Optional[tuple[float]] = Field(default=(12, 5))
    solartof_figsize: Optional[tuple[float]] = Field(default=(15, 5))
    solarrose_figsize: Optional[tuple[float]] = Field(default=(8, 8))
    evaporativecoolingpotential_figsize: Optional[tuple[float]] = Field(default=(15, 5))
    utcishadebenefit_figsize: Optional[tuple[float]] = Field(default=(22.8, 7.2))
    heatmaphistogram_figsize: Optional[tuple[float]] = Field(default=(15, 5))

    # colouring
    dbt_color: Optional[str] = Field(
        description="The color of the dry bulb temperature line.", default="#f04e23"
    )
    rh_color: Optional[str] = Field(
        description="The color of the relative humidity line.", default="#3a7ca5"
    )
    wbt_color: Optional[str] = Field(
        description="The color of the wet bulb temperature variable.", default="#3a7ca5"
    )
    dpt_color: Optional[str] = Field(
        description="The color of the dew point temperature variable.", default="#3a7ca5"
    )
    ws_color: Optional[str] = Field(
        description="The color of the wind speed variable.", default="Green"
    )
    # seasonality colouring
    winter_color: Optional[str] = Field(default="#4a71a6")
    summer_color: Optional[str] = Field(default="#fdcc08")
    spring_color: Optional[str] = Field(default="#89bba8")
    autumn_color: Optional[str] = Field(default="#baa75b")
    # categories
    utci_categories: Optional[CategoricalComfort] = Field(
        description="The UTCI categories to use.", default=UTCI_DEFAULT_CATEGORIES
    )
    # windrose
    windrose_cmap: Optional[str | Colormap] = Field(
        description="The colormap to use for the windrose plot.", default="YlGnBu"
    )
    windrose_directions: Optional[int] = Field(
        description="The number of directions to use for the windrose plot.", default=16
    )
    # file configurations
    overwrite: bool = Field(description="Overwrite existing outputs.", default=True)

    class Config:
        arbitrary_types_allowed = True
