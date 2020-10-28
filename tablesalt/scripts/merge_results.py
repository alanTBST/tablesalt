# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:26:56 2020

@author: alkj
"""

import pandas as pd


year = 2019

frames = []

res_cols = ['dsb', 'first', 'stog', 'movia', 'metro']
stats_cols = ['n_trips', 'n_period_cards', 'n_users', 'note']

for m in [1, 2, 3]:
    name_dict = {k:k+f'_model_{m}' for k in res_cols}
    model = pd.read_csv(f'takst_sjælland{year}_model_{m}.csv', low_memory=False)
    model = model.sort_values('NR')
    
    model.rename(columns=name_dict, inplace=True)
    
    if m > 1:
        model = model[name_dict.values()]
    else:
        new_order = \
            [x for x in model.columns if x 
             not in name_dict.values() and x not in stats_cols] + \
                stats_cols + list(name_dict.values())
        model = model[new_order]
    
    frames.append(model)
        
out = pd.concat(frames, axis=1)    
out.to_csv('kildefordeling_1_2_3.csv', index=False)