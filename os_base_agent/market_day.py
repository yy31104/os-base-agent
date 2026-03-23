from __future__ import annotations

from datetime import date

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    # Covers regular full-day NYSE holidays used by this strategy.
    # Does not include unscheduled special closures.
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_CAL = _NYSEHolidayCalendar()


def is_nyse_holiday(d: date) -> bool:
    hol = _NYSE_CAL.holidays(start=pd.Timestamp(d), end=pd.Timestamp(d))
    return len(hol) > 0


def is_nyse_trading_day(d: date) -> tuple[bool, str]:
    if d.weekday() >= 5:
        return False, "weekend"
    if is_nyse_holiday(d):
        return False, "nyse_holiday"
    return True, "trading_day"

