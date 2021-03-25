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

prisdf = gf.get_dataframe('msg* *pris', 'RejsePris > 0 & RejsePris <100',nmax=100)
prisdf.loc[:,'weekday'] = prisdf.Msgreportdate.dt.dayofweek
#%%
sns.boxplot(data=prisdf,y='RejsePris',x='weekday')
#%%
prisdf.hist(density=False)