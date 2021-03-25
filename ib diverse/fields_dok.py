# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:03:46 2021

@author: ib_hansen
"""
import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json 
import fnmatch

import get_fields

field_names = get_fields.get_field_names()

    
def fieldselect(pat='*'):
    with open('uni.pc', 'rb') as handle:
        uni = pickle.load(handle)
    patlist = pat.split(' ')
    out = {pat :uni.get(pat,pd.DataFrame([pat],index=[pat])).head(100)  for pat in patlist}
    # out =  {pat : df if type(df) == pd.DataFrame else get_fields.get_dataframe(pat,nmax=1).unique for pat,df in out0.items() }
    if 1:
        with pd.option_context('display.float_format','{:,.2f}'.format): 
            for name,df in out.items():
                try:
                    print(f'{name} : {df.head(2).index.to_list()}')
                except: 
                    print(f'\n\n{name:30}') 
    return out

xx = fieldselect('turngl Applicationtransactionsequencenu '+
       'KortnrKrypt Model StopPointNr NyUdfører msgraportdate')
