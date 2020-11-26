# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 09:51:04 2019

@author: alkj
"""
#standard imports

import os
import pickle
from itertools import groupby, chain
from functools import partial
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
#third party imports
import numpy as np
from tqdm import tqdm

# module imports
from tablesalt.running import WindowsInhibitor
from tablesalt import StoreReader
from tablesalt.common.io import mappers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.common import make_store
from tablesalt.topology.tools import TakstZones
from tablesalt.topology import ZoneGraph, ZoneSharer

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

CPU_USAGE = 0.4# %
DB_START_SIZE = 8 # gb


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
    store:
        the path to the h5 file
    region:

    """

    reader = StoreReader(store)
    stops = reader.get_data('stops')

    stops = stops[np.lexsort((stops[:, 1], stops[:, 0]))]
    # stops = stops[np.isin(stops[:, 0], list(ticket_tripkeys))]
    # print(len(set(stops[:, 0])))
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


def _get_store_num(store):

    st = store.split('.')[0]
    st = st.split('rkfile')[1]

    return st


def _convert_regional_to_sjælland():

    return


def _get_input(stop_dict, zone_dict, usage_dict, op_dict):

    for k, zone_sequence in zone_dict.items():
        yield k, zone_sequence, stop_dict[k], op_dict[k], usage_dict[k]


def chunk_shares(store, year, graph, region, zonemap, region_contractors):


    stopsd, zonesd, usaged, operatorsd = _load_store_data(
        store, region, zonemap, region_contractors
        )

    gen = _get_input(stopsd, zonesd, usaged, operatorsd)

    border_changes = {}
    model_one_shares = {}
    model_two_shares = {} # solo_zone_price


    # k, zones, stops, operators, usage = next(gen)
    for k, zones, stops, operators, usage in gen:

        sharer = ZoneSharer(graph, zones, stops, operators, usage)

        if sharer.border_trip:
            initial_zone_sequence = sharer.zone_sequence
            trip_shares = sharer.share()
            model_one_shares[k] = trip_shares
            final_zone_sequence = sharer.zone_sequence
            if initial_zone_sequence != final_zone_sequence:
                border_changes[k] = final_zone_sequence
            model_two_shares[k] = sharer.solo_zone_price()
        else:
            model_one_shares[k] = sharer.share()
            model_two_shares[k] = sharer.solo_zone_price()

    num = _get_store_num(store)
    #TODO ensure fp dir is created
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'borderzones',
        f'bzones{num}.pickle'
        )
    parent_dir = Path(fp).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(fp, 'wb') as f:
        pickle.dump(border_changes, f)

    return model_one_shares, model_two_shares

def main():
    """
    main function to create the operator
    shares for the data in the datastores
    """

    parser = TableArgParser('year')
    args = parser.parse()

    year = args['year']

    store_loc = find_datastores()
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']

    zones = TakstZones()
    zonemap = zones.stop_zone_map()

    # TODO into config
    region_contractors = {
        'hovedstaden': ['Movia_H', 'DSB', 'First', 'Stog', 'Metro'],
        'sjælland': ['Movia_H', 'Movia_S', 'Movia_V', 'DSB', 'First', 'Stog', 'Metro']
        }

    region = 'sjælland'
    graph = ZoneGraph(region=region)

    pfunc = partial(chunk_shares,
                    year=year,
                    graph=graph,
                    region=region,
                    zonemap=zonemap,
                    region_contractors=region_contractors)

    with Pool(round(os.cpu_count() * CPU_USAGE)) as pool:
        results = pool.imap(pfunc, stores)
        for model_one, model_two in tqdm(results, total=len(stores)):
            make_store(model_one, db_path, start_size=DB_START_SIZE)
            make_store(model_two, db_path + '_model_2', start_size=DB_START_SIZE)

if __name__ == "__main__":
    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
        main()
        INHIBITOR.uninhibit()
    else:
        main()

