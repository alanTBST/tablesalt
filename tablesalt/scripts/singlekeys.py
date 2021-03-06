# -*- coding: utf-8 -*-
"""
TBST Trafik, Bygge, og Bolig -styrelsen


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com

"""
import ast
import glob
import os
import pickle
from functools import partial
from itertools import groupby, chain
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import Set


import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm
# from turbodbc import connect, make_options

from tablesalt import StoreReader
from tablesalt.running import WindowsInhibitor
from tablesalt.topology import ZoneGraph
from tablesalt.topology.tools import TakstZones
from tablesalt.common.triptools import split_list
from tablesalt.common.connections import make_connection
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser


THIS_DIR = Path(__file__).parent
CPU_USAGE = 0.5

def _load_border_trips(year: int):
    "load and merge the dumped border trips"


    filedir = THIS_DIR / '__result_cache__' / f'{year}'/ 'borderzones'

    files = filedir.glob('*.pickle')
    borders = {}
    for file in tqdm(files, 'merging border trips'):
        with open(file, 'rb') as f:
            border = pickle.load(f)
            borders = {**borders, **border}
    return borders

# TODO into io
def helrejser_rabattrin(rabattrin, year):
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
        f"where [År] = {year} and [Manglende-check-ud] = 'Nej' and "
        f"Produktfamilie = '5' and [Rabattrin] = {rabattrin}"
        )
    # ops = make_options(
    #     prefer_unicode=True,
    #     use_async_io=True
    #     ) # , use_async_io=True
    with make_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        try:
            gen = cursor.fetchnumpybatches()
            try:
                out = set().union(*[set(batch['Turngl']) for batch in gen])
            except KeyError:
                try:
                    out = set().union(*[set(batch['turngl']) for batch in gen])
                except KeyError:
                    raise KeyError("can't find turngl")
        except AttributeError:
            gen = cursor.fetchall()

        out = set(chain(*gen))

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

def _separate_keys(short, long, _max, ringzones):

    ring = {}
    long_ = {}
    long_ring = {}

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

        if ringzones[(start_zone, end_zone)] != distance:
            continue
        if (start_zone, end_zone) not in long_:
            long_[(start_zone, end_zone)] = set()
        long_[(start_zone, end_zone)].add(k)

        if (start_zone, distance) not in long_ring:
            long_ring[(start_zone, distance)] = set()
        long_ring[(start_zone, distance)].add(k)

    # add 1 travelled zone to 2 travllled zones for short ring
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

def _determine_keys(stop_arr, stopzone_map, ringzones, bordertrips):

    zones = _map_zones(stop_arr, stopzone_map)

    borders = {k: v for k, v in bordertrips.items() if k in zones}
    zones.update(borders)
    region_trips = {
        'th': set(),
        'ts': set(),
        'tv': set(),
        'dsb': set()
        }

    for k, v in zones.items():
        if all(x < 1100 for x in v):
            region_trips["th"].add(k)
        elif all(1100 < x <= 1200 for x in v):
            region_trips["tv"].add(k)
        elif all(1200 < x < 1300 for x in v):
            region_trips["ts"].add(k)
        else:
            region_trips["dsb"].add(k)

    _max = _max_zones(zones, ringzones)
    short = {k: v for k, v in zones.items() if _max[k] <= 8}
    long = {k: v for k, v in zones.items() if _max[k] >= 9}
    tripkeys = _separate_keys(short, long, _max, ringzones)

    return tripkeys, region_trips

def _add_dicts_of_sets(dict1, dict2):

    return {key: dict1.get(key, set()) | dict2.get(key, set())
            for key in set(dict1) | set(dict2)}

def m_dicts(d1, d2):

    return {k: _add_dicts_of_sets(v, d2[k]) for k, v in d1.items()}

def merge_dicts(d1, d2):

    out = {}
    for k, v in d2.items():
        both_long = _add_dicts_of_sets(v['long'], d1[k]['long'])
        both_long_ring = _add_dicts_of_sets(
            v['long_ring'], d1[k]['long_ring']
            )

        both_short_dist = _add_dicts_of_sets(
            v['short_ring'],
            d1[k]['short_ring']
            )
        out[k] = {
            'long': both_long,
            'long_ring': both_long_ring,
            'short_ring': both_short_dist
            }

    return out

def _filter_operators(ticket_keys, operator_tripkeys: Set[int]):

    out = {}
    for k, v in ticket_keys.items():
        d = {
            k1: v1.intersection(operator_tripkeys)
            for k1, v1 in v.items()
            }
        out[k] = d

    return out

def _get_exception_stations(stops, *uic):

    tripkeys = stops[
        (np.isin(stops[:, 2], list(uic))) &
        (stops[:, -1] == 1)
        ][:, 0]

    return tripkeys


def _store_tripkeys(store, stopzone_map, ringzones, rabatkeys, bordertrips):

    reader = StoreReader(store)
    all_stops = reader.get_data('stops')
    all_stops = all_stops[np.isin(all_stops[:, 0], rabatkeys)]
    all_stops = all_stops[
        np.lexsort((all_stops[:, 1], all_stops[:, 0]))
        ]

    dr_byen = _get_exception_stations(all_stops, 8603311)
    dr_byen = all_stops[np.isin(all_stops[:, 0], dr_byen)]


    tick_keys, region_keys = _determine_keys(
        all_stops, stopzone_map, ringzones, bordertrips
        )

    dr_keys, dr_region_keys = _determine_keys(
        dr_byen, stopzone_map, ringzones, bordertrips
        )

    dr_keys['regions'] = dr_region_keys
    tick_keys['regions'] = region_keys

    return tick_keys, dr_keys

def _store_operator_tripkeys(store, ticket_keys, operators):

    reader = StoreReader(store)

    out = {}
    for operator in operators:
        operator_tripkeys = set(
            reader.get_data('stops', startswith=operator)[:, 0]
            )
        out[operator] = _filter_operators(
            ticket_keys, operator_tripkeys
            )

    return out


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


def _get_store_num(store):

    st = store.split('.')[0]
    st = st.split('rkfile')[1]

    return st


def _get_store_keys(store, stopzone_map, ringzones, operators, rabatkeys, year, bordertrips):

    tripkeys, dr_keys = _store_tripkeys(
        store, stopzone_map, ringzones, rabatkeys, bordertrips
        )
    op_tripkeys = _store_operator_tripkeys(
        store, tripkeys, operators
        )

    op_tripkeys['all'] = tripkeys
    op_tripkeys['dr_byen'] = dr_keys


    num = _get_store_num(store)
    fp = os.path.join(
        '__result_cache__',
        f'{year}',
        f'skeys{num}.pickle'
        )
    with open(fp, 'wb') as f:
        pickle.dump(op_tripkeys, f)


def _get_all_store_keys(stores, stopzone_map, ringzones, operators, rabatkeys, year):


    borders = _load_border_trips(year)

    pfunc = partial(_get_store_keys,
                    stopzone_map=stopzone_map,
                    ringzones=ringzones,
                    operators=operators,
                    rabatkeys=rabatkeys,
                    year=year,
                    bordertrips=borders)

    with Pool(round(os.cpu_count() * CPU_USAGE)) as pool:
        pool.map(pfunc, stores)


def _load_store_keys(filepath):

    with open(filepath, 'rb') as f:
        pack = pickle.load(f)
    return pack


def _merge_dicts(old, new, old_op, new_op, operators):

    out_all = m_dicts(old, new)
    out_operators = old_op.copy()
    for op in operators:
        out_operators[op] = m_dicts(
            out_operators[op], new_op[op]
            )

    return out_all, out_operators

def _gather_store_keys(lst_of_files, operators, nparts):

    initial = _load_store_keys(lst_of_files[0])
    out_all = initial['all']
    out_operators = {
        k: v for k, v in initial.items() if k != 'all'
        }

    for p in tqdm(lst_of_files[1:], f'merging store keys {nparts} parts'):
        keys = _load_store_keys(p)
        out = keys['all']
        opkeys = {k: v for k, v in keys.items() if k != 'all'}
        out_all, out_operators = _merge_dicts(
            out_all, out, out_operators, opkeys, operators
            )

    return out_all, out_operators

def _gather_all_store_keys(operators, nparts, year):


    lst_of_temp = glob.glob(
        os.path.join(
            THIS_DIR,
            '__result_cache__',
            f'{year}',
            '*.pickle'
            )
        )
    lst_of_temp = [x for x in lst_of_temp if 'skeys' in x]
    lst_of_lsts = split_list(lst_of_temp, wanted_parts=nparts)

    all_vals = []
    op_vals = []
    for lst in lst_of_lsts:
        out_all, out_operators = _gather_store_keys(lst, operators, nparts)
        all_vals.append(out_all)
        op_vals.append(out_operators)

    out_all = all_vals[0]
    out_operators = op_vals[0]
    for i in tqdm(range(len(all_vals)), 'merging subparts'):
        if i == 0:
            continue
        out = all_vals[i]
        opkeys = op_vals[i]
        out_all, out_operators = _merge_dicts(
            out_all, out, out_operators, opkeys, operators
            )

    for p in lst_of_temp:
        os.remove(p)

    return out_all, out_operators


def _get_rabatkeys(rabattrin, year):

    fp = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'preprocessed',
        f'rabat{rabattrin}trips.pickle'
        )
    try:
        with open(fp, 'rb') as f:
            rabatkeys = pickle.load(f)
    except FileNotFoundError:
        rabatkeys = helrejser_rabattrin(rabattrin, year)
        with open(fp, 'wb') as f:
            pickle.dump(rabatkeys, f)

    return rabatkeys


def _map_one_value(vals, res):
    t = (res.get(x) for x in vals)
    return tuple(x for x in t if x)


def _map_all(out_all, result_dict):

    new = {}
    for k, v in out_all.items():
        new[k] = {
            k1: _map_one_value(v1, result_dict) for k1, v1 in v.items()
            }

    return new

def _map_operators(out_operators, result_dict):

    new = {}
    for k, v in out_operators.items():
        new[k] = _map_all(v, result_dict)
        print(k)

    return new

def agg_nested_dict(node):
    if isinstance(node, tuple):
        return _aggregate_zones(node)
    else:
        dupe_node = {}
        for key, val in node.items():
            cur_node = agg_nested_dict(val)
            if cur_node:
                dupe_node[key] = cur_node
        return dupe_node or None

def _output_df(results, tick) -> pd.core.frame.DataFrame:

    frame = pd.DataFrame.from_dict(results, orient='index')
    frame = frame.fillna(0)
    frame = frame.reset_index()
    if 'ring' not in tick:
        frame = frame.rename(columns={
            'level_0': 'StartZone', 'level_1': 'DestinationZone'
            })
    else:
        frame = frame.rename(columns={
            'level_0': 'StartZone', 'level_1': 'n_zones'
            })
    col_order = frame.columns
    neworder = [x for x in col_order if x != 'n_trips']
    neworder.append('n_trips')
    frame = frame[neworder]

    return frame

def _write_results(rabattrin, year) -> None:

    dir_path = os.path.join(
        '__result_cache__',
        f'{year}',
        'single'
        )
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    fp = os.path.join(
        '__result_cache__',
        f'{year}',
        f'single_results_{year}_r{rabattrin}.pickle'
        )
    with open(fp, 'rb') as f:
        res = pickle.load(f)

    tmap = {'D**': 'DSB'}
    for start, tickets in res.items():
        for tick, results in tickets.items():
            df = _output_df(results, tick)
            name = tmap.get(start, start)
            fp = os.path.join(
                '__result_cache__',
                f'{year}',
                'single',
                f'start_{name}_{tick}_{year}_r{rabattrin}.csv'
                )
            df.to_csv(fp, index=False)


def _nzone_merge(resultdict):

    nzone = {k[1]: tuple() for k in resultdict}

    for k, v in resultdict.items():
        nzone[k[1]] = nzone[k[1]] + v

    return agg_nested_dict(nzone)


def _all_rabat_keys(year):

    all_rabat_levels = ()
    for i in range(0, 8):
        rabatkeys = tuple(_get_rabatkeys(i, year))
        all_rabat_levels += rabatkeys

    return all_rabat_levels

def _rabat_results(
        year,
        model,
        rabat_level,
        db_path,
        stores,
        stopzone_map,
        ringzones,
        wanted_operators
        ):

    single_fp = (THIS_DIR / '__result_cache__' /
                 f'{year}' / 'preprocessed' /
                 f'single_tripkeys_{year}_r{rabat_level}.pickle')


    if not single_fp.is_file():

        if rabat_level <= 7:
            rabatkeys = tuple(_get_rabatkeys(rabat_level, year))
        else:
            rabatkeys = _all_rabat_keys(year)

        _get_all_store_keys(
            stores,
            stopzone_map,
            ringzones,
            wanted_operators,
            rabatkeys,
            year
            )

        del rabatkeys
        nparts = 10
        out_all, out_operators = \
            _gather_all_store_keys(wanted_operators, nparts, year)

        out_operators['all'] = out_all
        with open(single_fp, 'wb') as file:
            pickle.dump(out_operators, file)
    else:
        with open(single_fp, 'rb') as f:
            res = pickle.load(f)
            out_all = res['all']
            out_operators = {k: v for k, v in res.items() if k != 'all'}

    print('finding results\n')
    all_wanted_keys = set()
    for k, v in out_all.items():
        for k1, v1 in v.items():
            all_wanted_keys.update(v1)

    result_dict = _get_trips(db_path, all_wanted_keys)

    print('loaded results\n')
    del all_wanted_keys

    all_results = _map_all(out_all, result_dict)
    short_all = _nzone_merge(all_results['short_ring'])
    long_all = _nzone_merge(all_results['long_ring'])
    all_results_ = agg_nested_dict(all_results)
    all_results_['paid_zones'] = {**short_all, **long_all}

    operator_results = _map_operators(out_operators, result_dict)
    operator_results_ = agg_nested_dict(operator_results)
    for k, v in operator_results.items():
        short_op = _nzone_merge(v['short_ring'])
        long_op = _nzone_merge(v['long_ring'])
        operator_results_[k]['paid_zones'] = {**short_op, **long_op}

    operator_results_['all'] = all_results_

    fp = Path(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'preprocessed',
        f'single_results_{year}_r{rabat_level}_model_{model}.pickle'
        )

    with open(fp, 'wb') as f:
        pickle.dump(operator_results_, f)


def main():

    parser = TableArgParser('year', 'model')
    args = parser.parse()

    year = args['year']
    model = args['model']

    store_loc = find_datastores()
    paths = db_paths(store_loc, year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']

    if model == 2:
        db_path = db_path + f'_model_{model}'


    ringzones = ZoneGraph.ring_dict('sjælland')
    stopzone_map = TakstZones().stop_zone_map()

    wanted_operators = [
        'Metro', 'D**', 'Movia_S', 'Movia_V', 'Movia_H'
        ]

    for rabat_level in [0, 1, 2]:
        _rabat_results(
        year,
        model,
        rabat_level,
        db_path, stores,
        stopzone_map,
        ringzones,
        wanted_operators
        )
    # _write_results(rabat_level, year) # csv results

if __name__ == "__main__":
    from datetime import datetime
    dt = datetime.now()
    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
        main()
        INHIBITOR.uninhibit()
    else:
        main()
    print(datetime.now() - dt)
