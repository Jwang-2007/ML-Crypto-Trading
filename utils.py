import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        sm.graphics.tsa.plot_acf(y,lags=lags, ax=acf_ax,alpha=.05)
        sm.graphics.tsa.plot_pacf(y,lags=lags, ax=pacf_ax,alpha=.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return fig

def corr_plot(responses,tech,tech_name):
    mergedDf = tech.merge(responses, left_index=True, right_index=True)
    mergedDf.dropna(inplace=True)
    corr = mergedDf.corr()
    sub_corr=corr.iloc[tech.shape[1]:, 0:tech.shape[1]]
    min_=sub_corr.min().min()
    max_=sub_corr.max().max()
    if max_>0:
        cmap_=sns.cubehelix_palette(100)
    else:
        cmap_=sns.color_palette("BuGn_r",100)
    fig, ax = plt.subplots(figsize=(10,10))         
    sns.heatmap(
        sub_corr, 
        vmin=min_, vmax=max_, center=(min_+max_)/2,
        cmap=cmap_,
        ax=ax)     
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    )
    ax.figure.savefig('./figures/corr_'+tech_name+'.png')
    return sub_corr