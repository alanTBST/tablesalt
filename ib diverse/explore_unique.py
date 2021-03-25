# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:22:45 2021

@author: ib_hansen

This module  a dict of frequency count for  fields in the delrejse dataset

The dict is retrieved from a pickle file created in create_unique_values 

"""

import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json 
import fnmatch


#%%
#%% Test the reading of uni 
with open('uni.pc', 'rb') as handle:
    unitest = pickle.load(handle)
    
def uniselect(uni,pat='*'):
    patlist = pat.split(' ')
    out = {name: df for name,df in uni.items()
          if any([fnmatch.fnmatch(name, pat) for pat in patlist])}

    with pd.option_context('display.float_format','{:,.2f}'.format): 
        for name,df in uni.items():
            print(f'\n\n{name:30} len={len(df)} \n{df}')

    return out

xx = uniselect(unitest,'*')

