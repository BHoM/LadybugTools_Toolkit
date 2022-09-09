from pathlib import Path

import pandas as pd
from ladybug.epw import EPW
from ladybugtools_toolkit.external_comfort.materials.materials import Materials

EPW_OBJ = EPW(Path(__file__).parent / "files" / "GBR_London.Gatwick.037760_IWEC.epw")
EPW_DF = pd.read_csv(
    Path(__file__).parent / "files" / "GBR_London.Gatwick.037760_IWEC.csv",
    parse_dates=True,
    index_col=0,
    header=0,
)
RES_FILE = Path(__file__).parent / "files" / "example.res"
SQL_FILE = Path(__file__).parent / "files" / "example.sql"
ILL_FILE = Path(__file__).parent / "files" / "example.ill"
PTS_FILE = Path(__file__).parent / "files" / "example.pts"

GROUND_MATERIAL = Materials.ASPHALT_PAVEMENT.value
SHADE_MATERIAL = Materials.FABRIC.value
IDENTIFIER = "pytest"
