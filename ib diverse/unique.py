# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:22:45 2021

@author: ib_hansen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
#%%
nmax = 100
res = get_freq('RejseAnalyse',nmax=nmax,normalize=0)
#%%   
nmax = 100
allnames = gf.get_field_names()
uni = {name : get_freq(name,nmax=nmax,normalize=0 ) 
       for i,name in enumerate(allnames) 
          if i < 50 and not name in {'RejsePris','Msgreportdate','tidsrabat'}} 

uni_len = {name : len(df) for name,df in uni.items()}
