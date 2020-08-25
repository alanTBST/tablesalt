# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 06:39:26 2020

@author: alkj
"""
import pickle

import pandas as pd

from tablesalt.topology import ZoneGraph

ringzones = ZoneGraph.ring_dict('sjælland')

with open('single_results.pickle', 'rb') as f:
    res = pickle.load(f)

tmap = {'D**': 'DSB'}
for start, tickets in res.items():
    for tick, results in tickets.items():
        df = pd.DataFrame.from_dict(results, orient='index')
        df = df.fillna(0)
        df = df.reset_index()
        if 'ring' not in tick:
            df = df.rename(columns={'level_0': 'StartZone', 'level_1': 'DestinationZone'})
            df['tup'] = df.loc[:, ('StartZone', 'DestinationZone')].apply(tuple, axis=1)
            df['ringdist'] = df['tup'].map(ringzones)
            if 'long' in tick:
                df = df.query("ringdist >= 9")
            else:
                df = df = df.query("ringdist <= 8")
        else:
            df = df.rename(columns={'level_0': 'StartZone', 'level_1': 'n_zones'})

        col_order = df.columns
        neworder = [x for x in col_order if x != 'n_trips']
        neworder.append('n_trips')
        df = df[neworder]
        name = tmap.get(start, start)
        if df.empty:
            print('...')
            break

        if name == 'Metro':
            df = df.query("StartZone <= 1004")
        df.to_excel(f'sjælland/start_{name}_{tick}_2019.xlsx')
