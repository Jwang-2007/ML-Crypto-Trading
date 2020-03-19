from talib import abstract
from collections import OrderedDict
import talib as tb
import pandas as pd

class technical_indicators(object):
    def __init__(self, data):
        self.data=data
        
    def get_idts(self,idts_name,para_list):
        func=abstract.Function(idts_name)
        func.input_arrays = self.data
        df=self.data[['close']]
        output_names=abstract.Function(idts_name).info['output_names']
        for x in para_list:
            pre = idts_name+str(x).strip("[]").replace(",","")+"_"
            col_names=[pre+x for x in output_names]
            func.parameters=OrderedDict(zip(list(func.parameters.keys()), x))
            #print(func.parameters)
            temp=func()
            if type(temp)!=pd.core.frame.DataFrame:
                temp=pd.DataFrame(temp,columns=col_names)
            else:
                temp.columns = col_names
            #print(temp)
            df=pd.concat([df,temp],axis=1)
        df.drop(columns=['close'],inplace=True)
        return df
    def get_info(self,idts_name):
        print(abstract.Function(idts_name).info)
    def get_price_idts(self):
        o = self.data['open'].values
        c = self.data['close'].values
        h = self.data['high'].values
        l = self.data['low'].values
        ta = pd.DataFrame()
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o
        ta.index=self.data.index
        return ta
    def get_composite_idts(self,idts_config):
        df=self.data[['close']]
        for idts_name,para_list in idts_config.items():
            df=pd.concat([df,self.get_idts(idts_name,para_list)],axis=1)
        df.drop(columns=['close'],inplace=True)
        return df   
        
    def default_idts(self):
        """
        Assemble a dataframe of technical indicator series for a single stock
        """
        o = self.data['open'].values
        c = self.data['close'].values
        h = self.data['high'].values
        l = self.data['low'].values
        v = self.data['volume'].astype(float).values
        # define the technical analysis matrix

        # Most data series are normalized by their series' mean
        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5) 
        ta['MA10'] = tb.MA(c, timeperiod=10)
        ta['MA20'] = tb.MA(c, timeperiod=20) 
        ta['MA60'] = tb.MA(c, timeperiod=60) 
        ta['MA120'] = tb.MA(c, timeperiod=120) 
        ta['MA5'] = tb.MA(v, timeperiod=5) 
        ta['MA10'] = tb.MA(v, timeperiod=10)
        ta['MA20'] = tb.MA(v, timeperiod=20) 
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) 
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) 
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] 
        ta['RSI'] = tb.RSI(c, timeperiod=14) 
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] 
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] 
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] 
        ta['AD'] = tb.AD(h, l, c, v) 
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) 
        ta['HT_DC'] = tb.HT_DCPERIOD(c) 
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o
        ta.index=self.data.index
        return ta    