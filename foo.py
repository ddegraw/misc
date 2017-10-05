import bt
import blpapi
import pandas as pd
import datetime as dt

#secs =['6758 JP Equity']#,'4452 JP Equity','4901 JP Equity']
#fld = ["VOLUME"]
#ind = "NKY Index"
#event = ["TRADE"]
#start = "2016-04-11T09:00:00"
#end = "2016-04-11T15:00:00"
#data = pd.DataFrame()
#for sec in secs:
#    df=blp.get_Ticks(sec,event,start,end)
#    data=data.combine_first(df)   
#sma = pd.rolling_mean(df,800)
#sma.rename(columns=lambda x: x + ' sma', inplace=True)
#plot = bt.merge(data, sma).plot(figsize=(15, 5))
#plt.show()

secu = "1332 JP Equity"
fld = ["VOLUME"]
ind = "NKY Index"
event = ["TRADE"]
sd = "2016-03-14T09:00:00"
ed = "2016-04-11T15:00:00"
iv = 5


bah = get_Bars(sec, fld_list, event_list, sdtime, edtime, barinterval)