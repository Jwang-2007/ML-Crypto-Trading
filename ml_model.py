import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from strategy_test import strategy_test
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from itertools import compress
import json
import joblib
import utils

def get_bbands_inputs(timeperiods):
    result=[]
    for x in timeperiods:
        if x<20:
            result.append([x,1.5,1.5,0])
        elif x<50:
            result.append([x,2,2,0])
        else:
            result.append([x,2.5,2.5,0])
    return result
def ml_config(period):
    #generating a bunch of tech inds based on windowsize
    windows=[period*n for n in range(1,7) if period*n<=90]
    time_periods=[[x] for x in windows]
    stoch_paras=[[x,x//2+1,0,x//2+1,0] for x in windows]
    config={'OBV':[[]],
        'ADXR':time_periods,'RSI':time_periods,'STOCH':stoch_paras,
        'NATR':time_periods,
        'SMA':time_periods,'BBANDS':get_bbands_inputs(windows),
        'BETA':time_periods,
       'LINEARREG_ANGLE':time_periods,'STDDEV':time_periods,
        'HT_TRENDMODE':[[]],'HT_DCPERIOD':[[]],'HT_SINE':[[]],'HT_DCPHASE':[[]]}
    return config

def feature_selection(X,Y,num_features=5):
    selector = SelectKBest(f_regression, k=num_features)
    selector.fit(X, Y)
    scores = -np.log10(selector.pvalues_)
    fig = plt.gcf()
    fig.set_size_inches(18, 6)
    plt.plot(X.columns,scores,'r-*',label='negative log p values')
    plt.xticks(rotation=90)
    plt.legend()
    return list(selector.get_support(indices=True))

def fit_techs(X,Y,tech_name,target,num=5):
    #tech_name:'RSI'
    #target:'log_rt_15'
    model = make_pipeline(
            SelectKBest(f_regression, k=num), MinMaxScaler(), RandomForestRegressor()
    )
    tt=get_split_index_by_window(X.shape[0])
    for x in tt:
        test_on_window(X,Y,x,tech_name,target,model)

def fit_compo_index(X,Y,est_name,target_name):
    if est_name=='rf':
        with open('./hypers/'+target_name+'_rf.json') as json_file:
            RF_best_params_ = json.load(json_file)
        rf = RandomForestRegressor(n_estimators=RF_best_params_['RFregressor__n_estimators'],
                           max_depth=RF_best_params_['RFregressor__max_depth'],
                           max_features=RF_best_params_['RFregressor__max_features'])
        model=Pipeline([
                    ('scaler',MinMaxScaler()), 
                    ('RFregressor',rf)
                    ])
    elif est_name=='svr':
        with open('./hypers/'+target_name+'_svr.json') as json_file:
            SVR_best_params_=json.load(json_file)
        model=Pipeline([
                ('scaler',MinMaxScaler()), 
                ('SVRregressor',SVR(gamma='scale',kernel='rbf',C=SVR_best_params_['SVRregressor__C'],
                                    epsilon=SVR_best_params_['SVRregressor__epsilon']))
                ])
    else:
        print('model '+est_name+' is not available.Train the hypers first.')
        return
    tt=get_split_index_by_window(X.shape[0])
    for x in tt:
        test_on_window(X,Y,x,est_name,target_name,model)        
def get_split_index(length,n_splits,samplesize=0.7):
    block_size=length//n_splits
    result=[]
    for i in range(n_splits):
        start = 0
        stop = i * block_size + block_size
        mid = int(samplesize*block_size) + i * block_size
        result.append((start,mid,stop))
    return result 
def get_split_index_shift(length,n_splits,samplesize=0.7):
    block_size=length//n_splits
    result=[]
    for i in range(n_splits):
        start = i * block_size
        stop = start + block_size
        mid = int(samplesize*(stop-start)) + start
        result.append((start,mid,stop))
    return result  

def get_split_index_by_window(length,window=90):
    n_splits=length//window
    initial=length%window
    result=[]
    for i in range(1,n_splits):
        start = 0
        mid =  initial + i*window
        stop = mid+window-1
        result.append((start,mid,stop))
    return result 

def calc_precision(y_true, y_pred):
    return np.sum((y_true>0.01) & (y_pred>0.01))/np.sum((y_pred>0.01)) if np.sum((y_pred>0.01))>0 else 0

def test_on_window(X,Y,indices,strg_name,target,clf_selected):
    #strg_name='RSI'
    #target='log_rt_15'
    start,mid,stop=indices
    X_train = X.iloc[start:mid, :]
    Y_train = Y.iloc[start:mid, :]

    X_test = X.iloc[mid:stop, :]
    Y_test = Y.iloc[mid:stop, :]

    clf_selected.fit(X_train,Y_train)  
    Y_pred=clf_selected.predict(X_test)

    fig_width = 12
    fig_height = 6
    fig = plt.figure(figsize=(fig_width,fig_height))
    layout = (1, 2)
    is_ax = plt.subplot2grid(layout, (0, 0))
    os_ax = plt.subplot2grid(layout, (0, 1))
    is_ax.scatter(clf_selected.predict(X_train),Y_train,c='r')
    os_ax.scatter(clf_selected.predict(X_test),Y_test,c='r')

    is_ax.set_title('in sample')
    os_ax.set_title('out of sample')

    is_ax.set_ylabel('real return')
    is_ax.set_xlabel('predicted return')
    os_ax.set_ylabel('real return')
    os_ax.set_xlabel('predicted return')

    filename='_'.join([strg_name,target,clf_selected.steps[-1][0],X.index[start].strftime("%Y%m%d"),X.index[stop].strftime("%Y%m%d")])
    fig.savefig('./results/'+filename+'.png')
    joblib.dump(clf_selected, './model/'+filename+'.joblib')
    #import strategy_test as st
    #in sample performance
    pred=pd.DataFrame({strg_name: clf_selected.predict(X_train)})
    pred.index=Y_train.index
    start=Y_train.index[0]
    end=Y_train.index[-1]
    rules={'action':'buy','y':target,'signal':strg_name}
    test=strategy_test(rules,pred,Y_train)
    test.model_testing(start,end)

    #out of sample performance
    pred=pd.DataFrame({strg_name: clf_selected.predict(X_test)})
    pred.index=Y_test.index
    start=Y_test.index[0]
    end=Y_test.index[-1]
    test=strategy_test(rules,pred,Y_test)
    test.model_testing(start,end)