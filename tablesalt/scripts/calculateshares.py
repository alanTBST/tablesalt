# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:51:04 2019

@author: alkj
"""
#standard imports


from itertools import groupby, chain
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
from tablesalt.revenue import multisharing
from tablesalt.topology.tools import TakstZones
from tablesalt.topology import (
    ZoneGraph,
    ZoneProperties,
    BORDER_STATIONS
    )


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

    filter_funcs = {
        'hovedstaden': lambda v: all(1000 < x < 1100 for x in v if x),
        'national': lambda v: all(x for x in v if x),
        'sjælland': lambda v: all(1000 < x < 1300 for x in v if x),
        'vestsjælland': lambda v: (_ for _ in v).throw(NotImplementedError('vestsjælland')),
        'fyn': lambda v: (_ for _ in v).throw(NotImplementedError('fyn')),
        'lolland': lambda v: (_ for _ in v).throw(NotImplementedError('lolland')),
        'nordjylland': lambda v: (_ for _ in v).throw(NotImplementedError('nordjylland')),
        'midtjylland': lambda v: (_ for _ in v).throw(NotImplementedError('midtjylland')),
        'sydjylland': lambda v: (_ for _ in v).throw(NotImplementedError('sydjylland'))
        }

    zone_dict = {k:v for k, v in zone_dict.items() if filter_funcs[region](v)}
    
    op_dict = _load_contractor_pack(store, region, region_contractors)
    op_dict = {k:v for k, v in op_dict.items() if k in zone_dict}

    stop_dict = {k:v for k, v in stop_dict.items() if k in op_dict}
    zone_dict = {k:v for k, v in zone_dict.items() if k in op_dict}
    usage_dict = {k:v for k, v in usage_dict.items() if k in op_dict}
    
    return stop_dict, zone_dict, usage_dict, op_dict

def chunk_shares(graph, store, region, zonemap, region_contractors):

    """
    Calculate the operator shares for trips in the
    rejsekort datastore

    parameters
    ----------
    rkstore:
        the path the the rejsekort h5 store

    :
        the path to the corresponding msgpack file
    """

    stop_dict, zone_dict, usage_dict, op_dict = _load_store_data(
        store, region, zonemap, region_contractors
        )

    singleops = _single_operator_assignment(
        graph, op_dict, zone_dict, stop_dict
        )
    multiops = _multi_operator_assignment(
        graph, op_dict, zone_dict, stop_dict, usage_dict
        )

    return {**singleops, **multiops}

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
    store = RK_STORES[87]
    DB_PATH = paths['calculated_stores']
    
    zones = TakstZones()
    zonemap = zones.stop_zone_map()
    
    region_contractors= {
        'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
        'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
        }
    
    region = 'sjælland'
    graph = ZoneGraph(region=region)
    rail_graph = ZoneGraph(region=region, mode='rail')
    bus_graph = ZoneGraph(region=region, mode='bus')

    for x in tqdm(RK_STORES, total=len(RK_STORES)):
        r = chunk_shares(graph, x, region, zonemap, region_contractors)
        make_store(r, DB_PATH)


    return r



# if __name__ == "__main__":

#     INHIBITOR = WindowsInhibitor()
#     INHIBITOR.inhibit()
#     main()
#     INHIBITOR.uninhibit()
