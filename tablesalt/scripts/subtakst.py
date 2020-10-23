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
import pandas as pd
from tqdm import tqdm

from tablesalt import StoreReader
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.topology.tools import TakstZones
from tablesalt.topology import ZoneGraph
from tablesalt.preprocessing.parsing import TableArgParser


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

# Put this in tools...it's in singlekeys
def _max_zones(operator_zones, ringdict):

    out = {}
    for k, v in operator_zones.items():
        current_max = 1
        start_zone = v[0]
        for zone in v[1:]:
            try:
                distance = ringdict[(start_zone, zone)]
                if distance > current_max:
                    current_max = distance
            except KeyError:
                break
        out[k] = current_max
    return out

def _load_data(store, stopzone_map, ringzones, rabatkeys):
    
    reader = StoreReader(store)
    allstops = reader.get_data('stops')
    allstops_rabat_zero = allstops[
        np.isin(allstops[:, 0], rabatkeys)
        ]
    allstops_mapped_zones = _map_zones(
        allstops_rabat_zero, stopzone_map
        )
    
    pensionstops = reader.get_data(
        'stops', passenger_type='pensioner'
        )
    peak_trips = reader.get_data(
        'stops', passenger_type='pensioner', 
        day=[0, 1, 2, 3, 4], hour=[7, 8, 15, 16]
        )    
    pensionstops = pensionstops[
        ~np.isin(pensionstops[:, 0], peak_trips[:, 0])
        ]
    
    pensionstops_mapped_zones = _map_zones(pensionstops, stopzone_map)
    pension_max_zones = _max_zones(pensionstops_mapped_zones, ringzones)
    threezones = {k for k, v in pension_max_zones.items() if v <= 3}
    pension_mapped_zones_three = {
        k: v for k, v in pensionstops_mapped_zones.items() if k in threezones
        }

    youthstops = reader.get_data(
        'stops', passenger_type='youth'
        )
    youth_mapped_zones = _map_zones(
        youthstops, stopzone_map
        )
    
    return (
        allstops_mapped_zones, 
        pensionstops_mapped_zones, 
        pension_mapped_zones_three,
        youth_mapped_zones
        )
        
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

def _get_trips(calculated_stores, tripkeys):

    with lmdb.open(calculated_stores, readahead=False) as env:
        out = {}
        with env.begin() as txn:
            for trip in tqdm(tripkeys):
                t = bytes(str(trip), 'utf-8')
                res = txn.get(t)
                if not res:
                    continue
                res = res.decode('utf-8')               
                if res not in ('operator_error', 'station_map_error'):
                    out[trip] = ast.literal_eval(res)
    return out
   
def main():
    
    parser = TableArgParser('year', 'rabattrin', 'model')
    args = parser.parse()
    rabat_level = args['rabattrin']
    model = args['model']

    year = args['year']
    
    store_loc = find_datastores(r'H://')
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']
    if model != 1:
        db_path = db_path + f'_model_{model}'    
    
    ringzones = ZoneGraph.ring_dict('sjælland')   
    rabatkeys = tuple(_get_rabatkeys(rabat_level, year))   

    stopzone_map = TakstZones().stop_zone_map()
       
    trips = {
        'alltrips': {
            'th': set(), 
            'ts': set(), 
            'tv': set(), 
            'city': set(), 
            'dsb': set()
             }, 
        'pension': {
            'th': set(), 
            'ts': set(), 
            'tv': set(), 
             }, 
        'pension_three': {
            'th': set(), 
            'ts': set(), 
            'tv': set(), 
             }, 
        'youth': {
            'th': set(),
            'ts': set(),
            'tv': set(),
             }
        }
    
    pfunc = partial(
        _load_data, 
        stopzone_map=stopzone_map, 
        ringzones=ringzones, 
        rabatkeys=rabatkeys
        )
    
    with Pool(7) as pool:
        results = pool.imap(pfunc, stores)       
        for res in tqdm(results, total=len(stores)):       
            alltrips, pension, pension_three, youth = res                
            for k, v in alltrips.items():       
                if all(x < 1100 for x in v):
                    trips['alltrips']["th"].add(k)
                    if all(x in (1002, 1002, 1003, 1004) for x in v):
                        trips['alltrips']["city"].add(k)             
                elif all(1100 < x <= 1200 for x in v):
                    trips['alltrips']["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips['alltrips']["ts"].add(k)
                else:
                    trips['alltrips']["dsb"].add(k)
            for k, v in pension.items():       
                if all(x < 1100 for x in v):
                    trips['pension']["th"].add(k)       
                elif all(1100 < x <= 1200 for x in v):
                    trips['pension']["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips['pension']["ts"].add(k)
                else:
                    pass
            for k, v in pension.items():       
                if all(x < 1100 for x in v):
                    trips['pension']["th"].add(k)       
                elif all(1100 < x <= 1200 for x in v):
                    trips['pension']["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips['pension']["ts"].add(k)
                else:
                    pass
            for k, v in pension_three.items():       
                if all(x < 1100 for x in v):
                    trips['pension_three']["th"].add(k)       
                elif all(1100 < x <= 1200 for x in v):
                    trips['pension_three']["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips['pension_three']["ts"].add(k)
                else:
                    pass
            for k, v in youth.items():       
                if all(x < 1100 for x in v):
                    trips['youth']["th"].add(k)       
                elif all(1100 < x <= 1200 for x in v):
                    trips['youth']["tv"].add(k)
                elif all(1200 < x < 1300 for x in v):
                    trips['youth']["ts"].add(k)
                else:
                    pass
    
    out = {}
    for k1, v1 in trips.items():
        sub = {}
        for k2, v2 in v1.items():
            sub[k2] = _get_trips(db_path, v2)
        out[k1] = sub
    
    test = {
        k1: {
            k2: get_user_shares(v2.values()) for k2, v2 in v1.items()
            } for k1, v1 in out.items() 
        }
    # test = {(i,j): test[i][j] for i in test.keys() for j in test[i].keys()}

    # return pd.DataFrame.from_dict(test, orient='index')
    return test
            
if __name__ == "__main__":
    results = main()       
        
    