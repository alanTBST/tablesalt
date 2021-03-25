# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:13:59 2021

@author: ib_hansen
"""

import pandas as pd
from pathlib import Path
import datetime
import gc
import fnmatch

path = Path(r'T:\Kollektiv trafik\Kollektiv trafik\TS503 Data\RejsekortSets\2019\sjælland\output') # path to output

outputdf = pd.read_csv(path / 'takst_sjaelland2019_model_1.csv',nrows = 1_000_000_000, encoding='iso-8859-1',
                       low_memory=False)
test = outputdf.rename(columns  = {'takstsÃ¦t':'Takstsat',})
field_names = outputdf.columns
uni  = {k : outputdf.loc[:,k].unique() for k in 'salgsvirksomhed produktgruppe takstsÃ¦t'.split()}
outputdf.query('takstsÃ¦t == "dlc')
