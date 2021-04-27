# -*- coding: utf-8 -*-
"""
Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com
"""

import ast
import os
import glob
import pickle
from functools import partial
from itertools import groupby, chain
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm


from tablesalt import StoreReader
from tablesalt.running import WindowsInhibitor
from tablesalt.topology import ZoneGraph
from tablesalt.topology.tools import TakstZones
from tablesalt.preprocessing.tools import find_datastores, db_paths

from tablesalt.scripts.singlekeys import _get_rabatkeys

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

CPU_USAGE = 0.6
def _determine_rabat_keys(year):
    """
    we want all epung trips that are r = 4,5,6,7

    Returns
    -------
    None.

    """

    wanted = set()

    for n in tqdm([4, 5, 6, 7], 'getting rabatkeys'):
        rkeys = _get_rabatkeys(n, year)
        wanted.update(rkeys)

    return wanted

def _load_border_trips(year: int):


    filedir = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'borderzones'
        )

    files = glob.glob(os.path.join(filedir, '*.pickle'))
    borders = {}
    for file in tqdm(files, 'merging border trips'):
        with open(file, 'rb') as f:
            border = pickle.load(f)
            borders = {**borders, **border}
    return borders


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

def _store_tripkeys(store, stopzone_map, rabatkeys, bordertrips):

    reader = StoreReader(store)
    all_stops = reader.get_data('stops')
    all_stops = all_stops[np.isin(all_stops[:, 0], rabatkeys)]
    all_stops = all_stops[
        np.lexsort((all_stops[:, 1], all_stops[:, 0]))
        ]
    zones = _map_zones(all_stops, stopzone_map)
    borders = {k: v for k, v in bordertrips.items() if k in zones}
    zones.update(borders)


    dset1 = tuple(k for k, v in zones.items() if 1001 in v)
    dset2 = tuple(k for k, v in zones.items() if 1002 in v)

    zone_one = tuple(k for k, v in zones.items() if set(v) == {1001})
    zone_one_three = tuple(k for k, v in zones.items() if set(v) == {1001, 1003})
    zone_one_three_four = tuple(k for k, v in zones.items() if set(v) == {1001, 1003, 1004})

    dset3 = zone_one + zone_one_three + zone_one_three_four

    return dset1, dset2, dset3

def _get_store_num(store):

    st = store.split('.')[0]
    st = st.split('rkfile')[1]

    return st


def _get_store_keys(store, stopzone_map, rabatkeys, year, bordertrips):

    dset1, dset2, dset3 = _store_tripkeys(
        store, stopzone_map, rabatkeys, bordertrips
        )

    d = {'zone1': dset1, 'zone2': dset2, 'zone134': dset3}
    num = _get_store_num(store)

    fp = os.path.join(
        '__result_cache__',
        f'{year}',
        f'zone2_{num}.pickle'
        )
    with open(fp, 'wb') as f:
        pickle.dump(d, f)


def _get_all_store_keys(stores, stopzone_map, rabatkeys, year):

    bordertrips = _load_border_trips(year)

    pfunc = partial(_get_store_keys,
                    stopzone_map=stopzone_map,
                    rabatkeys=rabatkeys,
                    year=year,
                    bordertrips=bordertrips)

    with Pool(round(os.cpu_count() * CPU_USAGE)) as pool:
        pool.map(pfunc, stores)

def _gather_all_store_keys(year):

    base = THIS_DIR.parent


    lst_of_temp = glob.glob(
        os.path.join(
            base,
            'scripts',
            '__result_cache__',
            f'{year}',
            '*.pickle'
        )
    )

    lst_of_temp = [x for x in lst_of_temp if 'zone2' in x]

    zone1 = tuple()
    zone2 = tuple()
    zone134 = tuple()
    for fp in tqdm(lst_of_temp, 'gathering store keys'):
        with open(fp, 'rb') as f:
            keys = pickle.load(f)
        zone1 += keys['zone1']
        zone2 += keys['zone2']
        zone134 += keys['zone134']

    # for p in lst_of_temp:
    #     os.remove(p)

    return zone1, zone2, zone134

def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values

    TODO put this in revenue module
    """
    n_trips = len(shares)
    multi = tuple(x for x in shares if isinstance(x[0], tuple))
    multi = tuple(chain(*multi))
    single = tuple(x for x in shares if isinstance(x[0], int))
    full = multi + single

    test_out = sorted(full, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)
    t = {key: sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    t = {k: v/totalzones for k, v in t.items()}
    t['n_trips'] = n_trips

    return t

def main(year):

    store_loc = find_datastores()
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']

    stopzone_map = TakstZones().stop_zone_map()

    rabatkeys = tuple(_determine_rabat_keys(year))

    # _get_all_store_keys(
    #     stores,
    #     stopzone_map,
    #     rabatkeys,
    #     year
    #     )

    zone1, zone2, zone134 = \
        _gather_all_store_keys(year)

    zone_1_results = _get_trips(db_path, zone1)
    zone_2_results = _get_trips(db_path, zone2)
    zone_134_results = _get_trips(db_path, zone134)

    a = _aggregate_zones(zone_1_results.values())
    b = _aggregate_zones(zone_2_results.values())
    c = _aggregate_zones(zone_134_results.values())


    return a, b, c

if __name__ == "__main__":
    a, b, c = main(2019)
    out = {}
    out['contains_zone1'] = a
    out['contains_zone2'] = b
    out['in_01_or_0103_or_0104'] = c
    import pandas as pd

    df = pd.DataFrame.from_dict(out, orient='index')
    df.to_csv('zone2_request.csv')
