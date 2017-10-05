# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation,TimeDistributedDense
from keras.layers import LSTM, Dropout


#sys.path.append('C:\\Anaconda2\\Lib\\site-packages\\blpfunctions')
import blpfunctions as blp
import datetime as dt
import pandas as pd
import numpy as np
from holidays_jp import CountryHolidays
import dateutil.rrule as RR
from pandas.tseries.offsets import *
from keras import callbacks

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

def volcurv(stock, event, edate, numdays, interval, fld_lst):

    volcurves = pd.DataFrame()
    fmt = "%Y-%m-%d" + 'T' + "%H:%M:%S"  #Assumes no milliseconds
    endDateTime = dt.datetime.strptime(edate, fmt)
    #skip SQ and holidays (Actually turns out cannot skip SQ days in blp call)
    #day1 = dt.datetime( int(edate[0:4]),1,1)  
    day1 = dt.datetime(2017,1,1)
    sq = list(RR.rrule(RR.MONTHLY,byweekday=RR.FR,bysetpos=2,dtstart=day1,until=endDateTime))
    hols = list(zip(*CountryHolidays.get('JP', int(edate[0:4])))[0])
    skipdays = sq + hols
    bday_jp = CustomBusinessDay(holidays=skipdays)  
    
    for i in range(numdays):
        endDateTime = dt.datetime.strptime(edate, fmt)
        startDateTime = endDateTime.replace(hour=9) - (i+1)*bday_jp
        endDateTime = startDateTime.replace(hour=15)
        sdate = startDateTime.strftime(fmt)
        endate = endDateTime.strftime(fmt)  
        output=blp.get_Bars(stock, event, sdate, endate, interval, fld_lst)
        output.rename(columns={'VOLUME':sdate},inplace=True)
        volcurves = volcurves.join(output,how="outer")

    #process the raw data into historical averages
    volcurves.rename(columns=lambda x: x[:10], inplace=True)
    timevect = pd.Series(volcurves.index.values)
    timeframet = timevect.to_frame()
    timeframet.columns =['date']
    timeframet.set_index(timevect,inplace="True")
    volcurves.set_index(timevect,inplace="True")#timezone hack
    timeframet['bucket'] = timeframet['date'].apply(lambda x: dt.datetime.strftime(x, '%H:%M:%S'))
    timeframet=timeframet.join(volcurves)
    volcurvesum=timeframet.groupby(['bucket']).sum()
    adv = volcurvesum.sum()/numdays
    volcurves = volcurvesum / volcurvesum.sum()
    volcurves = volcurves.cumsum()
    volcurves = volcurves.interpolate()
    volcurvesum = volcurvesum.interpolate()
    volcurvesum = volcurvesum.dropna(axis=1,how='all')
            
    return adv, volcurvesum.fillna(method='bfill'), volcurves.fillna(method='bfill')

stock = "2501 JT Equity"
fld = ["VOLUME"]
event = ["TRADE"]
ed1 = "2017-03-21T15:00:00"
ed = "2017-03-20T15:00:00"
iv = 5
numdays = 30

adv, rawcurve, volcurve = volcurv(stock,event,ed,numdays,iv,fld)
testadv, testraw, test = volcurv(stock,event,ed1,1,iv,fld)
df1=pd.DataFrame(volcurve.cumsum().T.stack())
#df1=pd.DataFrame(rawcurve.cumsum().T.stack())
#df1=pd.DataFrame(rawcurve.cumsum().T.stack())
df1.columns =['cumsum']


def _load_data(data, n_prev = 61):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 61):  
    """
    This just splits data to training and testing parts
    """   
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)
    

length_of_sequences = 61
in_out_neurons = 61
hidden_neurons = 61
bs = 61

#(X_train, y_train), (X_test, y_test) = train_test_split(df1[["cumsum"]], n_prev = length_of_sequences) 
(X_train, y_train), (X_test, y_test) = train_test_split(volcurve, n_prev = length_of_sequences) 
model = Sequential()  
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
#model.add(LSTM(hidden_neurons, input_dim=length_of_sequences, return_sequences=True))
model.add(Dropout(0.2))
#model.add(TimeDistributedDense(length_of_sequences))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(X_train, y_train, batch_size=bs, nb_epoch=15, validation_split=0.2, callbacks=[remote])  
     
#model.fit(X_train, y_train, batch_size=bs, nb_epoch=15, validation_data=(X_test, y_test), callbacks=[remote])     
     
predicted = model.predict(X_test) 
dataf =  pd.DataFrame(predicted[:1200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:1200]
dataf.plot(figsize=(15, 5))

#score = model.evaluate(X_test.as_matrix(), y_test, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)
