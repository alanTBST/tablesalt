# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import zipfile 
import os
from pathlib import Path
import datetime

path = Path(r'data/')


#%%
# zfile = Path(r'//tsdw03/h/Data/Rejsedata/2019/Delrejser/delrejse_2019_12_12.zip')
# zfile = Path(r'data/delrejse_2019_12_12.zip')

for zfile in path.glob('delrejse*.zip'): 
    print(f' Reading {zfile}')
    # to get a feel of the data 
    df_probe = pd.read_csv(zfile,encoding='iso-8859-1',nrows  = 100 )
    example_probe = df_probe.head().T
    
    
    # To get the field names in the delrejser data used in the rejsefordeling 
    # from delrejse_setup storeprocessing 
    "KortnrKrypt"
    
    col_dict = {
        'stoptime': [
            'turngl',
            'applicationtransactionsequencenu',
            'stoppointnr', 'model',
            'msgreportdate'
            ],
        'passengers': [
            'turngl', 'passagerantal1',
            'passagerantal2', 'passagerantal3',
            'passagertype1', 'passagertype2',
            'passagertype3', 'korttype'
            ],
        'price': [
            'turngl', 'rejsepris',
            'tidsrabat', 'zonerrejst'
            ],
        'contractor': [
            'turngl',
            'applicationtransactionsequencenu',
            'nyudfører', 'contractorid',
            'ruteid', 'fradelrejse',
            'tildelrejse'
            ]
        }
    relevant_names  = {c.upper() for l in col_dict.values() for c in l } | {"KortnrKrypt".upper()}
    GET_ALL_COLUMNS = True
    
    if GET_ALL_COLUMNS:
        relevant_columns = [c for c in df_probe.columns if c.upper()]
    else:
        relevant_columns = [c for c in df_probe.columns if c.upper() in relevant_names]
    
    #%% manage the datatypes 
    dtype_dict0 = {'turngl' : 'int64' , 'ContractorId':'category',
                   'RejsePris':'float64','tidsrabat':'float32',
                   'ZonerRejst' : 'category' ,
                   'Applicationtransactionsequencenu' : 'int64',
                   'KortnrKrypt'  : 'string'
                
                   }
    
    
    dtype_dict =  {v:dtype_dict0.get(v,'category') for v in relevant_columns if v != 'Msgreportdate' }
    
    # now read 
    df = pd.read_csv(zfile,encoding='iso-8859-1',nrows = 2_000_000_000,usecols = relevant_columns,
                     parse_dates = ['Msgreportdate'],dtype=dtype_dict,decimal=',',low_memory = True,
                     thousands='.' )
    #%%
    df.loc[:,'LastStopPointNr'] = df.StopPointNr.shift()
    
    dfnew = df.query('RejsePris > 0.0')
    dfnew.index = range(len(dfnew))
    example_dfnew  = dfnew.head().T
    example_df  = df.head().T
    weekday = dfnew.Msgreportdate.dt.weekday
    df.dtypes
    m1 = dfnew.memory_usage().sum()
    #%%
    
    with open(zfile.with_suffix('.ft'),'bw') as f:
        df.to_feather(f)
    
    # #%%
    # with open(filename,'br') as f:
    #     dftest = pd.read_feather(f)
    
