# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:17:45 2020

@author: alkj
"""
import ast
import glob
import os
import pickle
from operator import itemgetter
from functools import partial
from itertools import groupby, chain
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

from tablesalt import StoreReader
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.topology.tools import TakstZones

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

def _get_rabatkeys(rabattrin, year):

    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', 
        f'{year}',
        'preprocessed', 
        f'rabat{rabattrin}trips.pickle'
        )

    with open(fp, 'rb') as f:
        rabatkeys = pickle.load(f)

    return rabatkeys

def _map_zones(stop_arr, zonemap):
    # stop_arr must be sorted

    mapped_zones = {}
    stop_arr = stop_arr[np.lexsort((stop_arr[:, 1], stop_arr[:, 0]))]
    for key, grp in groupby(stop_arr, key=itemgetter(0)):
        zones = tuple(x[2] for x in grp)
        zones = tuple(zonemap.get(x, 0) for x in zones)
        if all(x > 0 for x in zones):
            mapped_zones[key] = zones

    return mapped_zones

def _load_data(store, rabatkeys, stopzone_map):
    
    stops = StoreReader(store).get_data('stops')
    stops = stops[np.isin(stops[:, 0], rabatkeys)]
    mapped_zones = _map_zones(stops, stopzone_map)
    
    return mapped_zones
    
 
def _get_trips(db, tripkeys):

    tripkeys_ = {bytes(str(x), 'utf-8') for x in tripkeys}

    out = {}
    with lmdb.open(db) as env:
        with env.begin() as txn:
            for k in tqdm(tripkeys_, 'loading trip results'):
                shares = txn.get(k)
                if shares:
                    try:
                        shares = shares.decode('utf-8')
                        out[int(k.decode('utf-8'))] = ast.literal_eval(shares)
                    except ValueError:
                        continue 

    return out

def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values
    """
    # TODO: import this from package
    test_out = sorted(shares, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)
    t = {key:sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    t = {k:v/totalzones for k, v in t.items()}

    return t


def get_user_shares(all_trips):
    # TODO: import this from package
    n_trips = len(all_trips)
    single = [x for x in all_trips if isinstance(x[0], int)]
    multi =  list(chain(*[x for x in all_trips if isinstance(x[0], tuple)]))
    all_trips = single + multi
    user_shares = _aggregate_zones(all_trips)
    user_shares['n_trips'] = n_trips

    return user_shares

def _get_trips(trips, calculated_stores):

    with lmdb.open(calculated_stores) as env:
        out = {}
        with env.begin() as txn:
            for trip in trips:
                t = bytes(trip, 'utf-8')
                res = txn.get(t)
                if not res:
                    continue
                res = res.decode('utf-8')               
                if res not in ('operator_error', 'station_map_error'):
                    out[trip] = ast.literal_eval(res)
    return out
   
def main():
    year = 2019
    store_loc = find_datastores(r'H://')
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']
    rabatkeys = tuple(_get_rabatkeys(0, 2019))
    
    stopzone_map = TakstZones().stop_zone_map()
    trips = {'th': set(), 
             'ts': set(), 
             'tv': set(), 
             'city': set(), 
             'dsb': set()}
    
    pfunc = partial(_load_data, 
                    rabatkeys=rabatkeys, 
                    stopzone_map=stopzone_map)
    
    with Pool(7) as pool:
        results = pool.imap(pfunc, stores)
        
        for res in tqdm(results, total=len(stores)):       
            for k, v in res.items():       
                if all(x < 1100 for x in v):
                    trips["th"].add(k)
                    if all(x in (1002, 1002, 1003, 1004) for x in v):
                        trips["city"].add(k)             
                elif all(1100 < x <= 1200 for x in v):
                    trips["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips["ts"].add(k)
                else:
                    trips["dsb"].add(k)
    
    out = {}
    for k, v in trips.items():
        out[k] = _get_trips(db_path, v)
    
    out = {k:get_user_shares(v.values()) for k, v in trips.items()}
    return out
            
if __name__ == "__main__":
    trips = main()            
