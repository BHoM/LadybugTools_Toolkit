"""Configuration options for the processing of climatic data from a pandas DataFrame"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..categorical.categories import UTCI_DEFAULT_CATEGORIES, CategoricalComfort


class MissingVariableException(Exception):
    """Custom exception raised when a required variable is missing."""


class Config(BaseModel):
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
    # file configurations
    overwrite: bool = Field(description="Overwrite existing outputs.", default=True)
