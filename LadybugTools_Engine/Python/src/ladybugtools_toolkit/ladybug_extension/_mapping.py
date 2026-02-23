import calendar

_MONTH_MAP = {calendar.month_abbr[i]: i for i in range(1, 13)}

_TIMESTEP_FREQUENCY_MAP = {
    1: "h",
    2: "30min",
    3: "20min",
    4: "15min",
    5: "12min",
    6: "10min",
    10: "6min",
    12: "5min",
    15: "4min",
    20: "3min",
    30: "2min",
    60: "min",
}
