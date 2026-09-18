"""Method to wrap for conversion of EPW to CSV file."""
# pylint: disable=C0415,E0401,W0703
import traceback
from ..logger import CONSOLE_LOGGER
from ladybugtools_toolkit.ladybug_extension.epw import epw_to_dataframe, EPW
from python_toolkit.bhom.decorators import bhom_wrapper

@bhom_wrapper.bhom_callable("epw_to_csv")
def epw_to_csv(epw_file: str, include_additional: bool, **kwargs) -> str:
    """Create a CSV file version of an EPW."""
    try:
        locator = kwargs.pop("epw_locator", None)
        if locator is not None:
            epw_file = locator(epw_file)

        epw = EPW(epw_file)
        df = epw_to_dataframe(epw=epw, include_additional=include_additional)
        csv_str = df.to_csv()
        return csv_str
    except Exception:
        CONSOLE_LOGGER.error("CSV file could not be created.", exc_info=1)
        return traceback.format_exc()
