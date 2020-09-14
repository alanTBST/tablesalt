# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:51:04 2019

@author: alkj
"""
#standard imports
import glob
import os
import sqlite3
import sys
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

def _trip_zone_properties(graph, zone_sequence, stop_sequence):
    """return a dictionary of zone properties"""
    return ZoneProperties(
        graph, zone_sequence, stop_sequence).property_dict

def _single_operator_assignment(graph, operators, zones, stops):

    """
    assign the zones travelled for single operator trips

    parameters
    -----------
    operators:
        the dictionary of tripkeys and operator id tuples

    zones:
        the dictionary of tripkeys and zone numbers

    stops:
        the dicionary of tripkeys and stopids

    stops, zones, usage are returned from _load_stop_store
    and operators is returned from _load_contractor_pack
    """

    single_operators = {k:v[0] for k, v in operators.items() if
                        len(set(v)) == 1}
    single_op_zones = {k:v for k, v in zones.items() if
                       k in single_operators}

    single_op_stops = {k:v for k, v in stops.items() if
                       k in single_operators}
    zone_properties = {}
    bad_keys = set()
    for k, v in single_op_zones.items():
        try:
            zone_properties[k] = _trip_zone_properties(graph, v, single_op_stops[k])
        except:
            bad_keys.add(k)
            continue

    shares = {
        k:(zone_properties[k]['total_travelled_zones'], single_operators[k])
        for k in single_operators if k in zone_properties
        }
    rev_op_map = {v: k for k, v in mappers['operator_id'].items()}

    shares = {
        k: (v[0], rev_op_map[v[1]].split('_')[0].lower()) for
        k, v in shares.items()
        }
    return shares

def _multi_operator_assignment(graph, operators, zones, stops, usage):
    """
    assign zone shares for multi operator trips

    parameters
    -----------
    operators:
        the dictionary of tripkeys and operator id tuples

    zones:
        the dictionary of tripkeys and zone numbers

    stops:
        the dicionary of tripkeys and stopids

    usage:
        the dicionary of tripkeys and usage ids,
        'Fi' = 1, 'Co' = 2, 'Su' = 3, 'Tr' = 4

    """
    multi_operators = {k:v for k, v in operators.items() if
                       len(set(v)) != 1}
    multi_op_zones = {k:v for k, v in zones.items() if
                      k in multi_operators}
    multi_op_stops = {k:v for k, v in stops.items() if
                      k in multi_operators}
    multi_op_usage = {k:v for k, v in usage.items() if
                      k in multi_operators}

    operator_legs = triptools.create_legs(multi_operators)
    usage_legs = triptools.create_legs(multi_op_usage)

    zone_properties = {}
    bad_keys = set()
    for k, v in multi_op_zones.items():
        try:
            zone_properties[k] = ZoneProperties(
                graph,
                v, multi_op_stops[k]
                ).property_dict
        except (TypeError, KeyError):
            bad_keys.add(k)
    bad_keys = {k:v for k, v in multi_op_zones.items() if k in bad_keys}

    return multisharing.share_calculation(
        operator_legs, usage_legs, zone_properties,
        BORDER_STATIONS
        )

def _load_contractor_pack(rkstore, region):
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
    reader = StoreReader(rkstore)
    contractors = reader.get_data('contractors')

    operator_ids = mappers['operator_id']

    contractor_filters = [
        operator_ids[x] for x in REGION_CONTRACTORS[region]
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

def _load_stop_store(rkstore, region, zonemap):
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

    reader = StoreReader(rkstore)
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
    stop_dict = {k:v for k, v in stop_dict.items() if k in zone_dict}
    usage_dict = {k:v for k, v in usage_dict.items() if k in zone_dict}

    return stop_dict, zone_dict, usage_dict

def chunk_shares(graph, store, region, zonemap):

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

    stop_dict, zone_dict, usage_dict = _load_stop_store(
        store, region, zonemap
        )
    op_dict = _load_contractor_pack(store, region)


    stop_dict = {k:v for k, v in stop_dict.items() if k in op_dict}
    zone_dict = {k:v for k, v in zone_dict.items() if k in op_dict}
    usage_dict = {k:v for k, v in usage_dict.items() if k in op_dict}

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
    DB_PATH = paths['calculated_stores']
    
    
    ZONES = TakstZones()
    ZONEMAP = ZONES.stop_zone_map()
    
    REGION_CONTRACTORS = {
        'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
        'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
        }
    
    region = 'sjælland'
    graph = ZoneGraph(region=region)

    for x in tqdm(RK_STORES, total=len(RK_STORES)):
        r = chunk_shares(graph, x, region)
        make_store(r, DB_PATH)


    return r



# if __name__ == "__main__":

#     INHIBITOR = WindowsInhibitor()
#     INHIBITOR.inhibit()
#     main()
#     INHIBITOR.uninhibit()
