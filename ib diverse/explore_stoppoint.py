# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:22:45 2021

@author: ib_hansen

Do stoppointnr match one to one stoppointid. 
this module extract to dublicate id's for each nr

At some point a little compressed. This is for memory reasons. 

"""

import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json 
import fnmatch

import get_fields as gf

if 1:
    unique_stoppoints = sorted(set(gf.get_dataframe('StopPointNr StopPointId', '',nmax=20).itertuples(index=False,name=None)))
    unique_stoppointsdf = pd.DataFrame(unique_stoppoints).rename(columns={1 : 'StopPointNr' ,0: 'StopPointId'})
    dublicate_StopPointNr = unique_stoppointsdf[unique_stoppointsdf.duplicated(['StopPointNr'],keep=False)]

if 0:  # if we want the counts, then nmax has to be <= 3
    lookat = gf.get_dataframe('StopPointNr StopPointId', '',nmax=3)
    df = pd.DataFrame(pd.Series(zip(*[lookat.StopPointNr,lookat.StopPointId])).value_counts().reset_index())
    df['StopPointNr'], df['StopPointId'] = zip(*df['index'])
    df = df.drop(columns='index').rename(columns={0 : 'count'})
    dublicate_StopPointNr = df[df.duplicated(['StopPointNr'],keep=False)]
    
