# -*- coding: utf-8 -*-
"""
TBST Trafik, Bygge, og Bolig -styrelsen


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com

"""
import ast
import glob
import os
import pickle
import sqlite3
import sys
from functools import partial
from itertools import groupby, chain
from multiprocessing import Pool, Process, Queue
from operator import itemgetter

import lmdb
from tqdm import tqdm
from turbodbc import connect, make_options

from tablesalt import StoreReader
from tablesalt.topology.tools import TakstZones
from tablesalt.topology import ZoneGraph
from tablesalt.common.triptools import split_list


def _find_datastores(start_dir=None):
    # TODO import this from preprocessing package

    if start_dir is None:
        start_dir = os.path.splitdrive(sys.executable)[0]
        start_dir = os.path.join(start_dir, '\\')
    for dirpath, subdirs, _ in os.walk(start_dir):
        if 'rejsekortstores' in subdirs:
            return dirpath
    raise FileNotFoundError("cannot find a datastores location")


def _make_db_paths(store_loc, year):

    usertrips_dir = os.path.join(
        store_loc, 'rejsekortstores', f'{year}DataStores',
        'dbs', 'user_trips_db'
        )
    kombi_dates_dir = os.path.join(
        store_loc, 'rejsekortstores', f'{year}DataStores',
        'dbs', 'kombi_dates_db'
        )

    kombi_valid_dates = os.path.join(
        store_loc, 'rejsekortstores', f'{year}DataStores',
        'dbs', 'kombi_valid_trips'
        )
    # this one exists from delrejsersetup.py
    tripcarddb = os.path.join(
        store_loc, 'rejsekortstores', f'{year}DataStores',
        'dbs', 'trip_card_db'
        )

    calc_db = os.path.join(
        store_loc, 'rejsekortstores', f'{year}DataStores',
        'dbs', 'calculated_stores'
        )

    return {'usertrips': usertrips_dir,
            'kombi_dates': kombi_dates_dir,
            'kombi_valid': kombi_valid_dates,
            'trip_card': tripcarddb,
            'calc_store': calc_db}


def _hdfstores(store_loc, year):

    return glob.glob(
        os.path.join(
            store_loc, 'rejsekortstores',
            f'{year}DataStores', 'hdfstores', '*.h5'
            )
        )


def helrejser_rabattrin(rabattrin):
    """
    return a set of the tripkeys from the helrejser data in
    the datawarehouse

    parameter
    ----------
    rabattrin:
        the value of the rabattrin
        int or list of ints
        default is None, returns all

    """

    query = (
        "SELECT Turngl FROM "
        "[dbDwhExtract].[Rejsedata].[EXTRACT_FULL_Helrejser_DG7] "
        "where [År] = 2019 and [Manglende-check-ud] = 'Nej' and "
        f"Produktfamilie = '5' and [Rabattrin] = {rabattrin}"
        )

    ops = make_options(
        prefer_unicode=True,
        use_async_io=True
        ) # , use_async_io=True
    with connect(
            driver='{SQL Server}',
            server="tsdw03",
            database="dbDwhExtract",
            turbodbc_options=ops
            ) as conn:

        cursor = conn.cursor()
        cursor.execute(query)
        gen = cursor.fetchnumpybatches()
        try:
            out = set().union(*[set(batch['Turngl']) for batch in gen])
        except KeyError:
            try:
                out = set().union(*[set(batch['turngl']) for batch in gen])
            except KeyError:
                raise KeyError("can't find turngl")
        cursor = conn.cursor()
        cursor.execute(query)
        gen = cursor.fetchnumpybatches()
        try:
            out = set().union(*[set(batch['Turngl']) for batch in gen])
        except KeyError:
            try:
                out = set().union(*[set(batch['turngl']) for batch in gen])
            finally:
                raise ValueError("can't find turngl")
    return {int(x) for x in out}

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

# for start_end
def _map_zones(operator_stops, zonemap):

    zonedict = {
        key: tuple(zonemap.get(x[2]) for x in grp)
        for key, grp in groupby(operator_stops, key=itemgetter(0))
        }
    return {k: v for k, v in zonedict.items() if all(x for x in v)}

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

def _separate_keys(short, long, _max):

    ring = {}
    long_ = {}
    long_ring = {}

    #TODO ensure long ring ends at max ringzone
    for k, v in short.items():
        start_zone = v[0]
        end_zone = v[-1]
        distance = _max[k]

        if (start_zone, distance) not in ring:
            ring[(start_zone, distance)] = set()
        ring[(start_zone, distance)].add(k)

    for k, v in long.items():
        start_zone = v[0]
        end_zone = v[-1]
        distance = _max[k]

        if start_zone == end_zone:
            continue
        if (start_zone, end_zone) not in long_:
            long_[(start_zone, end_zone)] = set()
        long_[(start_zone, end_zone)].add(k)

        if (start_zone, distance) not in long_ring:
            long_ring[(start_zone, distance)] = set()
        long_ring[(start_zone, distance)].add(k)

    # add one zone to two zones for short ring
    startzones = {k[0] for k in ring}
    for zone in startzones:
        one = ring.get((zone, 1), set())
        two = ring.get((zone, 2), set())
        ring[(zone, 2)] = one.union(two)

    analysis_tripkeys = {
        'short_ring': ring,
        'long': long_,
        'long_ring': long_ring
        }


    return analysis_tripkeys


def _add_dicts(dict1, dict2):

    return {key: dict1.get(key, set()) | dict2.get(key, set())
            for key in set(dict1) | set(dict2)}


def merge_dicts(d1, d2):

    out = {}
    for k, v in d2.items():
        both_long = _add_dicts(v['long'], d1[k]['long'])
        both_long_ring = _add_dicts(v['long_ring'], d1[k]['long_ring'])

        both_short_dist = _add_dicts(
            v['short_ring'],
            d1[k]['short_ring']
            )
        out[k] = {
            'long': both_long,
            'long_ring': both_long_ring,
            'short_ring': both_short_dist
            }

    return out

def _determine_keys(read_stops, stopzone_map, ringzones):

    zones = _map_zones(read_stops, stopzone_map)
    _max = _max_zones(zones, ringzones)
    short = {k: v for k, v in zones.items() if _max[k] <= 8}
    long = {k: v for k, v in zones.items() if _max[k] >= 9}
    tripkeys = _separate_keys(short, long, _max)

    return tripkeys

def _gather(stores, stopzone_map, ringzones):

    ops = ['Movia_H', 'Movia_S', 'Movia_V', 'Metro', 'D**']
    full_dict = {}
    for store in stores:
        reader = StoreReader(store)
        outdict = {}
        all_stops = reader.get_data('stops')
        tripkeys = _determine_keys(all_stops, stopzone_map, ringzones)
        outdict['all'] = tripkeys

        for operator in ops:
            stops = reader.get_data('stops', startswith=operator)
            tripkeys = _determine_keys(stops, stopzone_map, ringzones)
            outdict[operator] = tripkeys
        if not full_dict:
            full_dict = outdict
        full_dict = merge_dicts(full_dict, outdict)

    return full_dict

def _gather_all(stores, stopzone_map, ringzones):

    pfunc = partial(
        _gather,
        stopzone_map=stopzone_map,
        ringzones=ringzones
        )

    lst_of_stores = split_list(stores, wanted_parts=60)

    output = {}
    with Pool(6) as pool:
        res = pool.imap(pfunc, lst_of_stores)
        for r in tqdm(res, total=len(lst_of_stores)):
            if not output:
                output = r
            output = merge_dicts(output, r)

    return output


def _get_trips(db, tripkeys):


    tripkeys_ = {bytes(str(x), 'utf-8') for x in tripkeys}

    out = {}
    with lmdb.open(db) as env:
        with env.begin() as txn:
            for k in tripkeys_:
                shares = txn.get(k)
                if shares:
                    shares = shares.decode('utf-8')
                    out[int(k.decode('utf-8'))] = ast.literal_eval(shares)

    return out


def get_results(d, db_path, rabatkeys):

    all_keys = set.union(*d.values())
    all_keys = all_keys.intersection(rabatkeys)
    calc_vals = _get_trips(db_path, all_keys)

    out = {}
    for k, v in d.items():
        shares = []
        for tripkey in v:
            try:
                share = calc_vals[tripkey]
                shares.append(share)
            except KeyError:
                pass
        if shares:
            out[k] = tuple(shares)

    return {k: _aggregate_zones(v) for k, v in out.items()}


def _all_results(ddict, db_path, rabatkeys, q):

    long_res = get_results(
        ddict['long'], db_path, rabatkeys
        )
    long_d_res = get_results(
        ddict['long_ring'], db_path, rabatkeys
        )

    short_d_res = get_results(
        ddict['short_ring'],
        db_path, rabatkeys
        )

    q.put({'long': long_res,
           'long_ring': long_d_res,
           'short_ring': short_d_res,
           })

def main():


    ringzones = ZoneGraph().ring_dict('sjælland')
    stopzone_map = TakstZones().stop_zone_map()

    store_loc = _find_datastores(r'H://')
    db_dirs = _make_db_paths(store_loc, 2019)
    stores = _hdfstores(store_loc, 2019)

    db_path = db_dirs['calc_store']

    dicts = _gather_all(stores, stopzone_map, ringzones)

    try:
        with open('rabat0trips.pickle', 'rb') as f:
            rabatkeys = pickle.load(f)
    except FileNotFoundError:
        rabatkeys = helrejser_rabattrin(0)

    print('got rabatkeys')

    # all_queue = Queue(1)

    # p1 = Process(
    #     target=_all_results,
    #     args=(dicts['all'], db_path, rabatkeys, all_queue)
    #     )

    # movia_h_queue = Queue(1)
    # p2 = Process(
    #     target=_all_results,
    #     args=(dicts['Movia_H'], db_path, rabatkeys, movia_h_queue)
    #     )

    # metro_queue = Queue(1)

    # p3 = Process(
    #     target=_all_results,
    #     args=(dicts['Metro'], db_path, rabatkeys, metro_queue)
    #     )

    # dsb_queue = Queue(1)
    # p4 = Process(
    #     target=_all_results,
    #     args=(dicts['D**'], db_path, rabatkeys, dsb_queue)
    #     )

    # movia_s_queue = Queue(1)
    # p2 = Process(
    #     target=_all_results,
    #     args=(dicts['Movia_S'], db_path, rabatkeys, movia_s_queue)
    #     )


    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()

    results = {}
    # p1.join()

    # results['all'] = all_queue.get()

    # p2.join()
    # results['Movia_H'] = movia_h_queue.get()

    # p3.join()
    # results['Metro'] = metro_queue.get()

    # p4.join()
    # results['DSB'] = dsb_queue.get()


    # with open('single_results.pickle', 'wb') as f:
    #     pickle.dump(results, f)

    long_res = get_results(
        dicts['all']['long'], db_path, rabatkeys
        )
    long_d_res = get_results(
        dicts['all']['long_ring'], db_path, rabatkeys
        )

    short_d_res = get_results(
        dicts['all']['short_ring'],
        db_path, rabatkeys
        )

    results['all'] = {
        'long': long_res,
        'long_ring': long_d_res,
        'short_ring': short_d_res,
        }

    ops = ['Movia_H', 'Movia_S', 'Movia_V', 'Metro', 'D**']
    for x in ops:
        print('getting results for: ', x)

        long_res = get_results(
            dicts[x]['long'], db_path, rabatkeys
            )
        long_d_res = get_results(
            dicts[x]['long_ring'], db_path, rabatkeys
            )

        short_d_res = get_results(
            dicts[x]['short_ring'],
            db_path, rabatkeys
            )

        results[x] = {
            'long': long_res,
            'long_ring':long_d_res,
            'short_ring': short_d_res,
            }

    with open('single_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    # return dicts, res

if __name__ == "__main__":
    main()






