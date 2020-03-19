from collections import  OrderedDict
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class strategy_test(object):
    def __init__(self,rules,regressor,targets):
        """
        Set up a strategy env, feed in the data and initial status of the ledger

        :param env: market env collection
        :param initial: initial ledger status
        :param rules: controls over the strategy
        """
        self.regressor = regressor
        self.targets=targets
        self.rules = rules
    def stats_calc(self,tt,action=1):
        if action==-1:
            return OrderedDict([('sum', -1*tt.sum() if abs(tt.sum())>0 else 0), ('count',tt.count() if tt.count()>0 else 0), ('average', -1*tt.sum()/tt.count() if tt.count()>0 else 0)])    
        else:
            return OrderedDict([('sum', tt.sum() if abs(tt.sum())>0 else 0), ('count',tt.count() if tt.count()>0 else 0), ('average', tt.sum()/tt.count() if tt.count()>0 else 0)])
    
    def signal_stats(self,result,signal,pred,bd):
        stats=dict()
        if self.rules['action']=='buy':
            stats['FP']=self.stats_calc(result.loc[(result[signal]>=bd) & (result[pred] <0)][pred])
            stats['TP']=self.stats_calc(result.loc[(result[signal]>=bd) & (result[pred] >=0)][pred])
        else:
            stats['FP']=self.stats_calc(result.loc[(result[signal]<=bd) & (result[pred] >0)][pred],-1)
            stats['TP']=self.stats_calc(result.loc[(result[signal]<=bd) & (result[pred] <=0)][pred],-1)
        return stats
    def calc_avg_return(self,x):
        temp=(x[1]['FP']['sum']+x[1]['TP']['sum'])/(x[1]['FP']['count']+x[1]['TP']['count']) if (x[1]['FP']['count']+x[1]['TP']['count'])>0 else 0 
        if temp!=None:
            return temp
        else:
            return 0
    def calc_precision(self,x):
        temp=x[1]['TP']['count']/(x[1]['FP']['count']+x[1]['TP']['count']) if (x[1]['FP']['count']+x[1]['TP']['count'])>0 else 0
        if temp!=None:
            return temp
        else:
            return 0
    def model_testing(self, start, end):
        signal=self.rules['signal']
        pred=self.rules['y']
        result = pd.concat([self.regressor.loc[start:end,[signal]], self.targets.loc[start:end,[pred]]], axis=1, join='inner')
        result=result.dropna()
        l=result[signal].min()
        h=result[signal].max()
        if h-l<.0001:
            hypers=[l-0.1,(l+h)/2,h+0.1]
        else:
            hypers=[x for x in np.linspace(l,h,10)]+[0.01,0.02]
        temp=[]
        for x in hypers:
            bd=x
            temp.append((x,self.signal_stats(result,signal,pred,bd)))
        summary= pd.DataFrame({"boundary":hypers, 
                       "avg_return":[self.calc_avg_return(x) for x in temp],
                      "precision":[self.calc_precision(x) for x in temp]})
        summary.set_index('boundary',inplace=True)
        summary.sort_index(inplace=True)
        summary.dropna(inplace=True)
        fig_width = 12
        fig_height = 6
        fig, ax = plt.subplots(figsize=(fig_width,fig_height))
        ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

        width = 0.4
        print(summary)
        summary['precision'].plot(kind='bar', color='orange', ax=ax, width=width, position=1)
        summary['avg_return'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
        ax.set_title(' '.join(['signal:',signal,'action:',self.rules['action'],'prediction:',pred,start.strftime("%Y%m%d"),'to',end.strftime("%Y%m%d")]))
        ax.set_ylabel('precision')
        ax2.set_ylabel('avg. return')
        l=summary['avg_return'].min()
        h=summary['avg_return'].max()
        buffer=0.05*(h-l)
        if l*h<0:
            y1=abs(l-buffer)/(h+buffer)*1.05
            ax.set_ylim([-y1, 1.05])
            ax2.set_ylim([l-buffer,h+buffer])
        elif h<0: 
            h1=1.05*summary['precision'].max()
            h2=h*1.2
            ax2.set_ylim([h2,-h2])
            ax.set_ylim([-h1, h1])
        filename='_'.join([self.rules['action'],signal,pred,start.strftime("%Y%m%d"),end.strftime("%Y%m%d")])
        fig.savefig('./test/'+filename+'.png')
        summary.to_csv ('./results/'+filename+'.csv')
    def signal_testing(self, start, end):
        signal=self.rules['signal']
        pred=self.rules['y']
        result = pd.concat([self.regressor.loc[start:end,[signal]], self.targets.loc[start:end,[pred]]], axis=1, join='inner')
        result=result.dropna()
        l=int(result[signal].min())-1
        h=int(result[signal].max())+1
        hypers=[int(x) for x in np.linspace(l,h,20)]
        temp=[]
        for x in hypers:
            bd=x
            temp.append((int(x),self.signal_stats(result,signal,pred,bd)))
        summary= pd.DataFrame({"boundary":hypers, 
                       "avg_return":[(x[1]['FP']['sum']+x[1]['TP']['sum'])/(x[1]['FP']['count']+x[1]['TP']['count']) for x in temp],
                      "precision":[x[1]['TP']['count']/(x[1]['FP']['count']+x[1]['TP']['count']) for x in temp]})
        summary.set_index('boundary',inplace=True)
        summary.dropna(inplace=True)
        fig_width = 12
        fig_height = 6
        fig, ax = plt.subplots(figsize=(fig_width,fig_height))
        ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

        width = 0.4

        summary['precision'].plot(kind='bar', color='m', ax=ax, width=width, position=1)
        summary['avg_return'].plot(kind='bar', color='blue', ax=ax2, width=width, position=0)
        ax.set_title(' '.join(['signal:',signal,'action:',self.rules['action'],'prediction:',pred,start.strftime("%Y%m%d"),'to',end.strftime("%Y%m%d")]))
        ax.set_ylabel('precision')
        ax2.set_ylabel('avg. return')
        l=summary['avg_return'].min()
        h=summary['avg_return'].max()
        buffer=0.05*(h-l)
        if l*h<0:
            y1=abs(l-buffer)/(h+buffer)*1.05
            ax.set_ylim([-h1, h1])
            ax2.set_ylim([l-buffer,h+buffer])
        elif h<0: 
            h1=1.05*summary['precision'].max()
            h2=h*1.2
            ax2.set_ylim([h2,-h2])
            ax.set_ylim([-h1, h1])   
        filename='_'.join([self.rules['action'],signal,pred,start.strftime("%Y%m%d"),end.strftime("%Y%m%d")])
        fig.savefig('./test/'+filename+'.png')         