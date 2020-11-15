#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:11:53 2020


Create a dataset of the delrejser data and the
calculated results for those trips

@author: alan jones
@email: alkj@tbst.dk, alanksjones@gmail.com


"""

import glob
import os
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm


from tablesalt import StoreReader
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.common.io import mappers


THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent


def _load_single_tripkeys(year: int, rabattrin: int, model: int):
    
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', 
        f'{year}', 
        'preprocessed', 
        f'single_tripkeys_{year}_r{rabattrin}_model_{model}')
    
    with open(fp, 'rb') as file:
        tripkeys = pickle.load(file)
    
    return tripkeys


def _load_tripkeys():
    
    return 

def _load_single_trip_data(storepath, tripkeys):
    
    store = StoreReader(storepath)
    
    stops = store.get_data('stops')
    stops = stops[np.isin(stops[:, 0], list(tripkeys))]
    
    return stops

def _load_single_trips(storepaths, tripkeys):
    
    
    pfunc = partial(_load_single_trip_data, tripkeys=tripkeys)

    with Pool(10) as pool:       
        res = pool.map(pfunc, storepaths)

    
    return np.vstack(res)

def _load_results(dbpath, tripkeys):
    
    keys = {bytes(str(x), 'utf-8') for x in tripkeys}
    
    results = {}
    with lmdb.open(dbpath) as env:
        with env.begin() as txn:
            for tkey in tqdm(keys, 'getting trip results'):
                results[tkey] = txn.get(tkey)
    
    return results



def _merge_data_results(stopdata, results):
    
    
    stop_frame = pd.DataFrame(stopdata, columns=['tripkey', 'appseq', 'stopid', 'model'])
    
    res_data = pd.DataFrame.from_dict(results, orient='index')
    
    res_data = res_data.reset_index()
    res_data.columns = ['tripkey', 'shares']
    
    
    merge = pd.merge(stop_frame, res_data, on='tripkey', how='left')
    
    merge.loc[:, 'model'] = merge.loc[:, 'model'].map(mappers['model_dict'])
    
    return merge


def _load_pendler_trip_data():
    
    
    return 


def main():
    
    year = 2019
    rabat_level = 0
    model = 1
    ticket_group = 'single'
    ticket = (1005, 1033)
    
    tripkeys = _load_single_tripkeys(year, rabat_level, model)
    ticket_tripkeys = tripkeys['long'][ticket]
    
    paths = db_paths(find_datastores(), year)
    stores = paths['store_paths']
    
    stop_data = _load_single_trips(stores, ticket_tripkeys)
    

    results = _load_results(paths['calculated_stores'], ticket_tripkeys)
    
    return stop_data, results
    
    
    res = {}
    # for k, v in results.items():
    #     try:
    #         res[int(k.decode('utf-8')] = str(v, 'utf-8') 
    #     except TypeError:
    #         if 
    results = {
        int(k.decode('utf-8')): str(v, 'utf-8')
        for k, v in results.items()
        }
       
    merged = _merge_data_results(stop_data, results)
       
    
    stops = pd.read_csv(r'../resources/networktopodk/gtfs/stops.txt')
    
    stop_names = dict(zip(stops['stop_id'], stops['stop_name']))
    
    smaps = mappers['s_uic']
    smaps = [x - 90000 for x in smaps]
    smaps = {k: v for k, v in stop_names.items() if k in smaps}
    smaps = {k + 90000: v for k, v in smaps.items()}
    
    stop_names.update(smaps)
    
    merged['stop_name'] = merged.loc[:, 'stopid'].map(stop_names, na_action='ignore')
    
    merged.to_csv(f'tripsused_model_{model}_start_{ticket[0]}_paid_{ticket[1]}', index=False)
    
    return merged
    


if __name__ == "__main__":
    stop_data, results = main()
