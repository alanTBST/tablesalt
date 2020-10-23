# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:26:56 2020

@author: alkj
"""

import pandas as pd


model1 = pd.read_csv(r'takst_sjælland2019_model_1.csv', low_memory=False)
model1 = model1.sort_values('NR')

model2 = pd.read_csv('takst_sjælland2019_model_2.csv', low_memory=False)
model2 = model2.sort_values('NR')

model2 = model2[['NR', 'dsb', 'first', 'metro', 'movia', 'stog']]


res_cols = ['dsb', 'first', 'stog', 'movia', 'metro']
df = pd.merge(model1, model2, on='NR', suffixes=('_model_1', '_model_2'))

df = df[['NR', 'salgsvirksomhed', 'indtægtsgruppe', 'salgsår', 'salgsmåned',
       'takstsæt', 'produktgruppe', 'produktnavn', 'kundetype', 'salgsmedie',
       'betaltezoner', 'startzone', 'slutzone', 'valgtezoner', 'omsætning',
       'antal', 'n_trips', 'n_period_cards', 'n_users', 'note', 
       'dsb_model_1', 'first_model_1', 'stog_model_1',
       'metro_model_1', 'movia_model_1',
       'dsb_model_2', 'first_model_2', 
       'stog_model_2', 'movia_model_2','metro_model_2',
        ]]

df.to_csv('kildefordeling_1_2.csv', index=False)