# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:32:01 2016

@author: 4126694
"""

import datetime as dt
import pandas as pd
from pandas.tseries.offsets import *
from holidays_jp import CountryHolidays
import dateutil.rrule as RR

edate= "2016-04-28T15:00:00"
sd = dt.datetime( int(edate[0:4]),1,1)
ed = dt.datetime( int(edate[0:4]),12,31)

fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
endDateTime = dt.datetime.strptime(edate, fmt)


sq=list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=sd,until=ed))


hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])

off = hols+sq

bday_jp = CustomBusinessDay(holidays=off)
    
startDateTime = endDateTime.replace(hour=9) - 20*bday_jp
