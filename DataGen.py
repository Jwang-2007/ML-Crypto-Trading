import os
import sys
import requests
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

class DataGen(object):
    def __init__(self, source):
        self.source = source
        self.data=None
    def load(self):
        self.data = pd.read_csv('../data/webdata/Gemini_BTCUSD_d.csv',skiprows=1)
        self.data['mid'] = 0.5 * (self.data['High'] + self.data['Low'])
        self.data['Date']=self.data.apply(lambda row: datetime.strptime(row['Date'], '%Y-%m-%d'),axis=1)
        self.data.set_index('Date', inplace=True)
        self.data.sort_index(inplace=True)
        self.data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume BTC': 'volume'}, inplace=True) 
    def get_leading_log_rt(self,windowsizes):
        ta = pd.DataFrame()
        for i in windowsizes:
            #col_name = 'abs_rt_{}'.format(i)
            #data[col_name] = data['mid'].shift(i)-data['mid'] 
            col_name = 'log_rt_{}'.format(i)
            ta[col_name] = np.log(self.data['mid'].shift(-i) /self.data['mid'])
        ta.index=self.data.index
        return ta
    def get_leading_abs_rt(windowsizes):
        ta = pd.DataFrame()
        for i in windowsizes:
            #col_name = 'abs_rt_{}'.format(i)
            #data[col_name] = data['mid'].shift(i)-data['mid'] 
            col_name = 'abs_rt_{}'.format(i)
            ta[col_name] = self.data['mid'].shift(-i)/self.data['mid']-1
        ta.index=self.data.index  
        return ta
    def get_data(self):
        return deepcopy(self.data)  