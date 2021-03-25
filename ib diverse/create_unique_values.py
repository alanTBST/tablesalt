# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:22:45 2021

@author: ib_hansen

This module creates a dict of frequency count for  fields in the delrejse dataset

The dict is pickled into a file in current work directory called uni.pc 

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json 
import fnmatch
#%%
import get_fields as gf
def get_freq(field,nmax=1,normalize=True):
    this = gf.get_dataframe(field, '',nmax=nmax)
    if normalize:
        temp = (this.loc[:,field].value_counts(normalize=normalize)).astype('float64')*100
        fmt= '{:,.2f}%'.format
    else:
        temp = (this.loc[:,field].value_counts(normalize=normalize))
        fmt = '{:,}'.format
    res=pd.DataFrame(temp)
    with pd.option_context('display.float_format',fmt): 
        print(res)
    return res
#%% a test
nmax = 100
res = get_freq('RejseAnalyse',nmax=nmax,normalize=0)
#%% find  frequency for most fields  
nmax = 100
allnames = gf.get_field_names()
uni = {name : get_freq(name,nmax=nmax,normalize=0 ) 
       for i,name in enumerate(allnames) 
          if i < 50 and not name in {'Msgreportdate'}} 

uni_len = {name : len(df) for name,df in uni.items()}
#%% write pickle file 
with open('uni.pc', 'wb') as handle:
    pickle.dump(uni, handle)
#%% Test the reading of uni 
with open('uni.pc', 'rb') as handle:
    unitest = pickle.load(handle)
    
def uniselect(uni,pat='*'):
    patlist = pat.split(' ')
    fmt= '{:,.2f}'.format
    out = {name: df for name,df in uni.items()
          if any([fnmatch.fnmatch(name, pat) for pat in patlist])}

    with pd.option_context('display.float_format',fmt): 
        for name,df in uni.items():
            print(f'\n\n{name:30} len={len(df)} \n{df}')

    return out
xx = uniselect(unitest,'stop*')

#%%
#lookat = gf.get_dataframe('stop*', 'StopPointNr == "8603307"',nmax=100)
lookat = gf.get_dataframe('stop*', '',nmax=1)
b = pd.Series(zip(*[lookat.StopPointNr,lookat.StopPointId]))
df = pd.DataFrame(b.value_counts().reset_index())
df['StopPointNr'], df['StopPointId'] = zip(*df['index'])
df = df.drop(columns='index').rename(columns={0 : 'count'})
dublicate_StopPointNr = df[df.duplicated(['StopPointNr'],keep=False)]
