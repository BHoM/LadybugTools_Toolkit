import os
import numpy as np
import pandas as pd
import calendar
import dataframe_image as dfi
import math
from typing import List

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from IPython.display import display
from ladybugtools_toolkit.plot.colormaps_class import UTCIColorScheme
from ladybugtools_toolkit.plot.colormaps_classes import UTCIColorSchemes



# NOTE: Helper function
def dynamic_grouping(totalTimeCount: int, groupingSpan: int = 1) -> List[int]:
    # NOTE: The current groupingSpan variable if it is a list of numbers, it has to add up to totalTimeCount
    if type(groupingSpan) is not int and sum(groupingSpan) != totalTimeCount and len(groupingSpan) > 1:
        raise ValueError(
            "Variable groupingSpan is not a singular value or sum up to " + str(totalTimeCount) +" hours! Change the list value passed or change it to a singular hour." 
        )
    
    dynamicGroups = []
    if type(groupingSpan) is int:
            for i in range(totalTimeCount):
                dynamicGroups.append(math.floor(i / groupingSpan))
    else:
        for i in range(len(groupingSpan)):
            dynamicGroups.extend((np.ones(groupingSpan[i])*i))
            
    return dynamicGroups



def utci_tabulated_class_color_delta(collection: HourlyContinuousCollection, collection_base: HourlyContinuousCollection,
                         hourSpan: int = 1, monthSpan: int = 1,
                         UTCIColor: UTCIColorScheme = UTCIColorSchemes.UTCI_Original,
                         Excel_Path: str = None, Image_Path: str = None
) -> pd.io.formats.style.Styler: 
    """Create a styled dataframe showing the annual hourly UTCI values associated with custom Hour Span and Month Span.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        collection_base (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object that represents the base value to compare to.
        hourSpan (int/[int], optional):
            A custom hour span for table. Can be a single value or a list of values. Default is 1.
        monthSpan (int/[int], optional):
            A custom month span for table. Can be a single value or a list of values. Default is 1.
        UTCIColor (UTCIColorScheme, optional):
            Option to use customized UTCI color scheme object. Default is None.
        Excel_Path (str, optional):
            Full path for the exported Excel File, Default is None
        Image_Path (str, optional):
            Full path for the exported Image File, Default is None
    return:
        A styled dataframe object
    """

    # Check collection type
    if not isinstance(UTCIColor, UTCIColorScheme):
        raise ValueError(
            "UTCIColor is not a UTCI color scheme object and cannot be used in this plot."
        )
        
    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )
    
    # Construct series
    series = to_series(collection)
    df_series = pd.DataFrame(series, columns=[series.name])
    df_series.index = pd.to_datetime(df_series.index, format='%Y-%m-%d %H:%M:%S')
    
    # Hour and Month groupby pattern
    hour_group = dynamic_grouping(24,hourSpan)
    h_groups = np.unique(hour_group) 

    month_group = dynamic_grouping(12,monthSpan)
    m_groups = np.unique(month_group)

    # Compute base on grouping
    df_series = df_series.groupby([df_series.index.month, df_series.index.hour]).mean().unstack(level=0)
    df_series = df_series.groupby(hour_group, axis=0).mean().groupby(month_group,axis=1).mean().reset_index(drop=True)

    # Create value for background color control
    np_value = df_series.copy().to_numpy()
    np_value = [list(pd.cut(row, bins=[-100] + UTCIColor.UTCI_LEVELS + [200], labels=list(range(len(UTCIColor.UTCI_LABELS))))) for row in np_value]
    np_value.reverse()
    
    # Format for table dispaly
    # Construct series
    series_base = to_series(collection_base)
    df_series_base = pd.DataFrame(series_base, columns=[series_base.name])
    df_series_base.index = pd.to_datetime(df_series_base.index, format='%Y-%m-%d %H:%M:%S')
    
    # Compute base on grouping
    df_series_base = df_series_base.groupby([df_series_base.index.month, df_series_base.index.hour]).mean().unstack(level=0)
    df_series_base = df_series_base.groupby(hour_group, axis=0).mean().groupby(month_group,axis=1).mean().reset_index(drop=True)

    # in this case calculate Celsius Average  
    celsiusAverage = df_series.to_numpy()
    fahrenheitAverage = (celsiusAverage * 1.8 + 32)
    celsiusAverage_base = df_series_base.to_numpy()
    
    df_series = df_series.astype('string')
    for i in range(len(h_groups)):
        for j in range(len(m_groups)):
            df_series.at[i,j] = str(celsiusAverage[i][j].round(1)) + " | "  + str(fahrenheitAverage[i][j].round(1)) + " ("  + str((celsiusAverage[i][j]-celsiusAverage_base[i][j]).round(1)) + ")"
    df_series = df_series.iloc[::-1]

    # Set index & column names
    monthFrom = []
    monthTill = []
    for i in range(len(m_groups)):
        monthFrom.append(np.where(month_group == m_groups[i])[0][0]+1)
        monthTill.append(np.where(month_group == m_groups[i])[0][-1]+1)
    monthFrom = list(map(lambda x: calendar.month_abbr[x], monthFrom))
    monthTill = list(map(lambda x: calendar.month_abbr[x], monthTill))
    monthCol = list(map(lambda X: (X[0]+" to "+ X[1]), list(zip(monthFrom,monthTill))))
    
    # potential helper function
    hourFrom = []
    hourTill = []
    for i in range(len(h_groups)):
        hourFrom.append(np.where(hour_group == h_groups[i])[0][0])
        hourTill.append(np.where(hour_group == h_groups[i])[0][-1]+1)
    hourFrom = [f"{x:02}" for x in hourFrom]
    hourTill = [f"{x:02}" for x in hourTill]
    hoursIndex = list(map(lambda X: (X[0]+ ":00 to " + X[1] + ":00"), list(zip(hourFrom,hourTill))))
    hoursIndex.reverse()

    df_series.columns = monthCol
    df_series.index = hoursIndex

    # Style DF output
    styled = df_series.style.background_gradient(axis=None, cmap=UTCIColor.UTCI_COLORMAP, gmap=np_value, vmin=1, vmax=len(UTCIColor.UTCI_LEVELS)-1, text_color_threshold=1)

    # Save to EXCEL / IMG
    if Excel_Path:
        styled.to_excel(Excel_Path + ".xlsx")
        print("Excel Saved to " + Excel_Path + ".xlsx!")
    if Image_Path:
        dfi.export(styled, Image_Path + ".jpeg")
        print("Image Saved to " + Image_Path + ".jpeg!")

    return styled