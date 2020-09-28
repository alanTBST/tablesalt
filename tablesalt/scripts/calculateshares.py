# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:51:04 2019

@author: alkj
"""
#standard imports

import os
from itertools import groupby, chain
from functools import partial
from multiprocessing import Pool
from operator import itemgetter

#third party imports
import numpy as np
from tqdm import tqdm

# module imports
from tablesalt.running import WindowsInhibitor
from tablesalt import StoreReader
from tablesalt.common.io import mappers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.common import triptools, make_store
from tablesalt.topology.tools import TakstZones
from tablesalt.topology import ZoneGraph, ZoneSharer



# NOTE: SAVE BORDER TRIP STARTZONES
def proc_contractors(contrpack):
    """return the contractors dict as an array"""
    arr_length = len(tuple(chain(*contrpack.values())))
    arr = np.zeros(shape=(arr_length, 4), dtype=np.int64)
    i = 0
    for k, v in contrpack.items():
        for record in v:
            arr[i, 0] = k
            arr[i, 1] = record[0]
            arr[i, 2] = record[1]
            arr[i, 3] = record[2]
            i += 1
    return arr

def _load_contractor_pack(store, region, region_contractors):
    """
    load and process the operator information from
    msgpack file

    parameters
    ----------
    rkpack:
        the msgpack file path corresponding to the rkstore

    filter_type:
        the region filter for the operators
        currenlty only 'hovedstaden' is supported and
        is the default value
    """
    reader = StoreReader(store)
    contractors = reader.get_data('contractors')

    operator_ids = mappers['operator_id']

    contractor_filters = [
        operator_ids[x] for x in region_contractors[region]
        ]

    contractors = proc_contractors(contractors)
    bad_ops = contractors[:, 0][
        ~np.isin(contractors[:, 2], contractor_filters)
        ]

    contractors = contractors[~np.isin(contractors[:, 0], bad_ops)]
    contractors = contractors[
        np.lexsort((contractors[:, 1], contractors[:, 0]))
        ]

    op_dict = {key:tuple(x[2] for x in grp) for key, grp in
               groupby(contractors, key=itemgetter(0))}

    return op_dict

def _load_store_data(store, region, zonemap, region_contractors):
    """
    load the stop data from the h5 file and create
    the stop, zone and usage dicts

    parameters
    ----------
    rkstore:
        the path to the h5 file
    filter_type:
        filter the trips in the store by the takstzone region
        default is 'hovedstaden'

        currently only ['hovedstaden', 'national'] supported;
        ['sjælland', 'vestsjælland', 'fyn', 'lolland',
        'nordjylland', 'midtjylland', 'sydjylland']
        will be implemented at a later date and raise
        NotImplementedError

    """

    reader = StoreReader(store)
    stops = reader.get_data('stops')

    stops = stops[np.lexsort((stops[:, 1], stops[:, 0]))]
    

    
    usage_dict = {
        key: tuple(x[3] for x in grp) for key, grp in
        groupby(stops, key=itemgetter(0))
        }
    usage_dict = {
        k: v for k, v in usage_dict.items() if
        v[0] == 1 and v[-1] == 2
        }
    stop_dict = {
        key: tuple(x[2] for x in grp) for key, grp in
        groupby(stops, key=itemgetter(0)) if key in usage_dict
        }
    zone_dict = {
        k: tuple(zonemap.get(x) for x in v) for
        k, v in stop_dict.items()
        }    
    zone_dict = {
        k:v for k, v in zone_dict.items() if 
        all(x for x in v) and all(1000 < y < 1300 for y in v)
        }
    
    op_dict = _load_contractor_pack(store, region, region_contractors)
    op_dict = {k:v for k, v in op_dict.items() if k in zone_dict}

    stop_dict = {k:v for k, v in stop_dict.items() if k in op_dict}
    zone_dict = {k:v for k, v in zone_dict.items() if k in op_dict}
    usage_dict = {k:v for k, v in usage_dict.items() if k in op_dict}
    
    return stop_dict, zone_dict, usage_dict, op_dict


def _get_input(stop_dict, zone_dict, usage_dict, op_dict):
    
    for k, zone_sequence in zone_dict.items():
        yield k, zone_sequence, stop_dict[k], op_dict[k], usage_dict[k]

def chunk_shares(store, graph, region, zonemap, region_contractors):

    stopsd, zonesd, usaged, operatorsd = _load_store_data(
        store, region, zonemap, region_contractors
        )
    
    gen = _get_input(stopsd, zonesd, usaged, operatorsd)
    
    shares = {}
    for k, stops, zones, usage, operators in gen:    

        sharer = ZoneSharer(graph, stops, zones, usage, operators)
        shares[k] = sharer.share()
    
    fixed = {}
    for k, v in shares.items():
        if isinstance(v[0], int):
            fixed[k] = v[0], v[1].lower().split('_')[0]
        elif isinstance(v[0], tuple):
            fixed[k] = tuple((x[0], x[1].lower().split('_')[0]) for x in v)
        else:
            fixed[k] = v
    return fixed

def main():
    """
    main function to create the operator
    shares for the data in the datastores
    """

    parser = TableArgParser('year')
    args = parser.parse()
    
    year = args['year']

    store_loc = find_datastores(r'H://')
    paths = db_paths(store_loc, year)
    RK_STORES = paths['store_paths']
    DB_PATH = paths['calculated_stores']
    
    zones = TakstZones()
    zonemap = zones.stop_zone_map()
    
    region_contractors= {
        'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
        'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
        }
    
    region = 'sjælland'
    graph = ZoneGraph(region=region)

    pfunc = partial(chunk_shares, 
                    graph=graph, 
                    region=region, 
                    zonemap=zonemap, 
                    region_contractors=region_contractors)
    
    with Pool(os.cpu_count() - 2) as pool:
        results = pool.imap(pfunc, RK_STORES)
        for r in tqdm(results, total=len(RK_STORES)):       
            make_store(r, DB_PATH)

if __name__ == "__main__":

    INHIBITOR = WindowsInhibitor()
    INHIBITOR.inhibit()
    main()
    INHIBITOR.uninhibit()
