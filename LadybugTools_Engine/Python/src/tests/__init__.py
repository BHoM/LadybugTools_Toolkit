from pathlib import Path

from ladybug.epw import EPW

EPW_OBJ = EPW(Path(__file__).parent / "files" / "GBR_London.Gatwick.037760_IWEC.epw")
RES_FILE = Path(__file__).parent / "files" / "example.res"
SQL_FILE = Path(__file__).parent / "files" / "example.sql"
ILL_FILE = Path(__file__).parent / "files" / "example.ill"
PTS_FILE = Path(__file__).parent / "files" / "example.pts"
