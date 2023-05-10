import os
import numpy as np
import pandas as pd
import calendar
import dataframe_image as dfi

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.temperature import UniversalThermalClimateIndex
from ladybugtools_toolkit.ladybug_extension.datacollection.to_series import to_series
from IPython.display import display

def utci_tabulated_style(collection: HourlyContinuousCollection, hourSpan: int = [1], monthSpan: int = [1], 
IP: bool = False, EXCEL: bool = False, IMG: bool = False, analysis_period: AnalysisPeriod = AnalysisPeriod()
): 
    """Create a styled dataframe showing the annual hourly UTCI values associated with custom Hour Span and Month Span.

    Args:
        collection (HourlyContinuousCollection):
            A ladybug HourlyContinuousCollection object.
        hourSpan ([int], optional):
            A custom hour span for table. Can be a single value or a list of values. Default is 1.
        monthSpan ([int], optional):
            A custom month span for table. Can be a single value or a list of values. Default is 1.
        IP (bool, optional):
            Convert data to IP unit. Default is True.
        EXCEL (bool, optional):
            Save styled table to an excel. Default is False.
        IMG (bool, optional):
            Save styled table to an image. Default is False.
        analysis_period (AnalysisPeriod, optional):
            A ladybug analysis period.
    """
    if not isinstance(collection.header.data_type, UniversalThermalClimateIndex):
        raise ValueError(
            "Collection data type is not UTCI and cannot be used in this plot."
        )
    # Convert data to IP unit
    if IP:
        collection = collection.to_ip()

    # Construct series
    series = to_series(collection)
    df0 = pd.DataFrame(series, columns=[series.name])
    df0.index = pd.to_datetime(df0.index, format='%Y-%m-%d %H:%M:%S')
    
    # Hour Span
    # NOTE: The current hourSpan variable if it is a list of hours, it has to add up to 24
    if sum(hourSpan) != 24 and len(hourSpan) > 1:
        raise ValueError(
            "Variable hourSpan is not a singular value or sum up to 24 hours! Change the list value passed or change it to a singular hour." 
        )
    
    # NOTE: The current monthSpan variable if it is a list of months, it has to add up to 12
    if sum(monthSpan) != 12 and len(monthSpan) > 1:
        raise ValueError(
            "Variable monthSpan is not a singular value or sum up to 12 month! Change the list value passed or change it to a singular hour." 
        )

    if len(hourSpan) == 1:
        df0 = df0.resample(str(hourSpan[0]) + 'H').mean()
        rowN = 24 // hourSpan[0]
    else:
        orderedSpan = hourSpan.copy()
        orderedSpan.sort()
        start = 0
        rollingHour = np.zeros(shape=(len(hourSpan),12))
        hours = []
        for i in range(len(hourSpan)):
            dfTemp = df0.copy()
            dfTemp = dfTemp.resample(str(hourSpan[i]) + 'H').mean()
            dfTemp = dfTemp.groupby([dfTemp.index.month, dfTemp.index.hour]).mean().unstack(level=0)
            hours.append(start)
            rollingHour[i] = dfTemp.loc[start]
            start += hourSpan[i]
        df0 = pd.DataFrame(rollingHour)
        df0.columns = np.arange(1,13,1)
        rowN = len(hourSpan)

    # Month Span
    if len(monthSpan) == 1:
        monthSpan = [monthSpan[0]] * int(12 / monthSpan[0])
        colN = int(12 / monthSpan[0])
    # print(monthSpan)
    
    grouping = []
    for i in range(len(monthSpan)):
        grouping.extend((np.ones(monthSpan[i])*i).round(0))
    df0 = df0.groupby(grouping,axis=1).mean().round(1).reset_index(drop=True)
    colN = len(monthSpan)

    # Calculate Celsius & Fahrenheit Average  
    celsiusAverage = df0.to_numpy()
    fahrenheitAverage = (celsiusAverage * 1.8 + 32).round(1)
    df0 = df0.astype('string')
    for i in range(rowN):
        for j in range(colN):
            #df0.at[i,j] = str(celsiusAverage[i][j]) + " | " + str(fahrenheitAverage[i][j])
            df0.at[i,j] = str(celsiusAverage[i][j])
    maxInd, maxCol = [x[0] for x in np.unravel_index([np.argmax(df0)], df0.shape)]
    df0 = df0.iloc[::-1]

    # Set index & column names
    groups = np.unique(grouping)
    monthFrom = []
    monthTill = []
    for i in range(len(groups)):
        monthFrom.append(np.where(grouping == groups[i])[0][0]+1)
        monthTill.append(np.where(grouping == groups[i])[0][-1]+1)
    monthNameFirst = list(map(lambda x: calendar.month_abbr[x], monthFrom))
    monthNameSecond = list(map(lambda x: calendar.month_abbr[x], monthTill))
    monthCol = list(map(lambda X: (X[0]+" to "+ X[1]), list(zip(monthNameFirst,monthNameSecond))))
    

    hoursTill = [f"{x:02}" for x in hours[1:]]
    hours = [f"{x:02}" for x in hours]
    if len(hoursTill) < len(hours):
        hoursTill.append(str(24))
    hoursIndex = list(map(lambda X: (X[0]+ ":00 to " + X[1] + ":00"), list(zip(hours,hoursTill))))
    hoursIndex.reverse()
    # outsideIndex = [df0.index.tolist()[x] for x in outsidePeriod]

    df0.columns = monthCol
    df0.index = hoursIndex

    # Style DF output
    styled = df0.style.applymap(lambda x: ("background-color: #f7c1aa" if ((float(x[:4])>28)) 
                                        else ("background-color: #d3d317" if float(x[:4])>26 
                                                else "background-color: #2eb349")))
    """
    # Set analysis_period
    if analysis_period.st_hour != 0:
        startInd = hours.index(analysis_period.st_hour)
    else:
        startInd = analysis_period.st_hour
    if analysis_period.end_hour != 23:
        endInd = hours.index(analysis_period.end_hour)
    else:
        endInd = analysis_period.end_hour
    
    outsidePeriod = [*range(0,startInd, 1)]
    if endInd < len(hours):
        outsidePeriod = outsidePeriod + [*range(endInd, len(hours))]

    # Style DF output
    #  Potential future dev NOTE: add analysis period with MONTH with shading
    #  Potential future dev NOTE: use UTCI color scheme
    if analysis_period.end_hour != 23 and analysis_period.st_hour != 0:
        styled = df0.style.applymap(lambda x: ("background-color: #f7c1aa" if ((float(x[:4])>28)) 
                                        else ("background-color: #d3d317" if float(x[:4])>26 
                                                else "background-color: #2eb349")))
        #.set_properties(subset=((df0.index[x] for x in outsidePeriod),df0.columns),**{"opacity": "0.5"})
        #.set_properties(#subset=((df0.index[startInd]),df0.columns),**{"border-top":"2px dashed black"})
        #.set_properties(#subset=((df0.index[endInd]),df0.columns),**{"border-bottom":"2px dashed black"})
        #.set_properties(#subset=(df0.index[maxInd], df0.columns[maxCol]),**{"border": "2px solid red"})
    elif analysis_period.end_hour != 23:
        styled = df0.style.applymap(lambda x: ("background-color: #f7c1aa" if ((float(x[:4])>28)) 
                                        else ("background-color: #d3d317" if float(x[:4])>26 
                                                else "background-color: #2eb349")))
        #.set_properties(subset=((df0.index[x] for x in outsidePeriod),df0.columns),**{"opacity": "0.5"})
        #.set_properties(subset=((df0.index[endInd]),df0.columns),**{"border-bottom":"2px dashed black"})
        #.set_properties(subset=(df0.index[maxInd], df0.columns[maxCol]),**{"border": "2px solid red"})
    elif analysis_period.st_hour != 0:
        styled = df0.style.applymap(lambda x: ("background-color: #f7c1aa" if ((float(x[:4])>28)) 
                                        else ("background-color: #d3d317" if float(x[:4])>26 
                                                else "background-color: #2eb349")))
        #.set_properties(subset=((df0.index[x] for x in outsidePeriod),df0.columns),**{"opacity": "0.5"})
        #.set_properties(subset=((df0.index[startInd]),df0.columns),**{"border-top":"2px dashed black"})
        #.set_properties(subset=(df0.index[maxInd], df0.columns[maxCol]),**{"border": "2px solid red"})
    else:
        styled = df0.style.applymap(lambda x: ("background-color: #f7c1aa" if ((float(x[:4])>28)) 
                                        else ("background-color: #d3d317" if float(x[:4])>26 
                                                else "background-color: #2eb349")))
        #.set_properties(subset=(df0.index[maxInd], df0.columns[maxCol]),**{"border": "2px solid red"})
    """

    # Save to EXCEL / IMG
    if EXCEL:
        styled.to_excel('UTCI_Tabulated_Styled.xlsx')
        print("Excel Saved!")
    if IMG:
        dfi.export(styled,'UTCI_Tabulated_Styled.jpeg')
        print("Image Saved!")

    return styled