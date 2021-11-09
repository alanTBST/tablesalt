# -*- coding: utf-8 -*-
"""
What does it do?
================

Aggregates pendler tickets:

    - by chosen pendlerkombi zones
    - by the number of paid to
    - by zones matching the DOT zone relations

These are then dumped in a __result_cache__ folder for later processing
by the salesoutput.py script


USAGE
=====

To run the script for the year 2019 using model 1 (standard zonework model)

    python ./path/to/tablesalt/tablesalt/scripts/pendlerkeys.py -y 2019
    -p /path/to/PeriodeProdukt.csv -z /path/to/Zoner.csv -m 1

where -p is the path to the period product data provided by rejsekort,
-z is the path to the product zones data provided by rejsekort
and -m is the model to run (1 --> standard zonework model,
2 --> solo zoner price model, 3 --> equal operator model)

"""
import ast
from collections import defaultdict
import os
import pickle
import pkg_resources
from datetime import datetime
from itertools import groupby, chain
from functools import lru_cache
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import AnyStr, Set, Tuple, Dict

import lmdb
import msgpack
import pandas as pd
from tqdm import tqdm

from tablesalt.running import WindowsInhibitor
from tablesalt import StoreReader
from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser

THIS_DIR = Path(__file__).parent

def get_zone_combinations(udata) -> Set[Tuple[int, ...]]:
    """Get all the chosen zone combinations of pendler kombi users

    :param udata: userdata loaded from users.PendlerInput class
    :type udata: Dict
    :return: distinct valid zone combinations of all users.
    :rtype: Set[Tuple[int], ...]
    """

    zone_set = set()
    for _, seasons in udata.items():
        for _id, card_info in seasons.items():
            if card_info['zones'] not in zone_set:
                zone_set.add(card_info['zones'])
    # THIS IS ONLY FOR SJÆLLAND < 1300
    return {x for x in zone_set if all(y < 1300 for y in x)}

def get_users_for_zones(
    udict: PendlerKombiUsers,
    zone_set: Set[Tuple[int, ...]]
    ) -> Tuple[Dict, Dict]:
    """For each distinct zone combination, find their users and season passes

    :param udict: the user dictionary for pendler kombi users.
    :type udict: PendlerKombiUsers
    :param zone_set: distinct valid zone combinations of all users
    :type zone_set: Set[Tuple[int], ...]
    :return: the users/seasonpasses for each set of valid zones, summary statistics
    :rtype: Tuple[Dict, Dict]
    """

    zone_set_users = {}
    statistics = {}

    for zone_combo in zone_set:
        combo_users = udict.subset_zones(zones=zone_combo)
        card_seasons = {
            k: tuple(v.keys()) for
            k, v in combo_users[0].items()
            }
        zone_set_users[zone_combo] = card_seasons
        statistics[zone_combo] = {'n_users': combo_users[1]}
        statistics[zone_combo]['n_period_cards'] = combo_users[2]

    return zone_set_users, statistics


def proc_user_data(udata, combo_users):
    """


    Parameters
    ----------
    udata : dict
        userdata loaded from users.PendlerInput class.

    combo_users : dict
        zone_set_users.

    Returns
    -------
    dict
        filtered dict of userdata with only valid times.

    """
    outdata = {}
    for k, v in udata.items():
        for k1, v1 in v.items():
            outdata[(k, k1)] = v1['start'].date(), v1['end'].date()

    users_ = list(combo_users.values())
    card_season = set(chain(*[[tuple((k, x) for x in v) for
                               k, v in user.items()] for user in users_]))
    card_season = set(chain(*card_season))

    return {k:v for k, v in outdata.items() if k in card_season}



def load_valid(valid_kombi_store: str) -> Dict:
    """
    Load the valid kombi user trips

    Returns
    -------
    valid_kombi : dict
        DESCRIPTION.

    """
    with lmdb.open(valid_kombi_store, readahead=False) as env:
        valid_kombi = {}
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                valid_kombi[k.decode('utf-8')] = ast.literal_eval(v.decode('utf-8'))

    return {ast.literal_eval(k): v for k, v in valid_kombi.items()}

@lru_cache(2*16)
def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)

def _wrap_price(store):
    """

    :param store: DESCRIPTION
    :type store: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    # TODO: rather use partial
    return StoreReader(store).get_data('price')

def get_zeros(p):
    df = pd.DataFrame(p[:, (0, 1)],
                      columns=['tripkey', 'price']).set_index('tripkey')
    df = df.groupby(level=0)['price'].transform(max)
    df = df[df == 0]
    return tuple(set(df.index.values))

def proc(store):

    price = _wrap_price(store)

    return get_zeros(price)

def find_no_pay(stores, year):

    fp = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'preprocessed',
        'zero_price.pickle'
        )

    try:
        with open(fp, 'rb') as file:
            out = pickle.load(file)
    except FileNotFoundError:
        out = set()
        with Pool(os.cpu_count() - 2) as pool:
            results = pool.imap(proc, stores)
            for res in tqdm(results,
                            'finding trips inside zones',
                            total=len(stores)):
                out.update(set(res))
        with open(fp, 'wb') as f:
            pickle.dump(out, f)
    return out

def assert_internal_zones(zero_travel_price, zone_combo_trips):

    return {k: v.intersection(zero_travel_price) for k, v in zone_combo_trips.items()}

def trips_to_zone_combination(season_trips, zone_combo_users):

    zone_combo_trips = {}
    for zones, userseasons in tqdm(
            zone_combo_users.items(),
            'matching trips to zone combinations'
            ):
        useasons = set(chain(*[tuple((k, x) for x in v) for k, v in userseasons.items()]))
        zone_trips = set(chain(*{v for k, v in season_trips.items() if k in useasons}))
        zone_combo_trips[zones] = zone_trips

    return zone_combo_trips

def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values
    """
    # TODO: import this from package
    test_out = sorted(shares, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)
    t = {key: sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    t = {k: v/totalzones for k, v in t.items()}

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


def n_operators(share_tuple):

    return len({x[1] for x in share_tuple})

def pendler_reshare(share_tuple):

    n_ops = n_operators(share_tuple)

    return tuple((1/ n_ops, x[1]) for x in share_tuple)

def get_zone_combination_shares(tofetch, db_path: str, model: int):

    final = {}
    errors = {'operator_error', 'station_map_error', 'rk_operator_error', 'no_available_trip'}
    with lmdb.open(db_path) as env:
        with env.begin() as txn:
            for combo, trips in tqdm(tofetch.items(),'fetching combo results',
                                     total=len(tofetch)):
                all_trips = {}
                for trip in trips:
                    t = bytes(str(trip), 'utf-8')
                    res = txn.get(t)
                    if not res:
                        continue
                    res = res.decode('utf-8')
                    if res not in errors:
                        val = ast.literal_eval(res)
                        # if model != 3:
                        all_trips[trip] = val
                        # else:
                        #     all_trips[trip] = pendler_reshare(val)
                combo_result = get_user_shares(all_trips.values())
                final[combo] = combo_result

    return final

# =============================================================================
# aggregation by paid zones
# =============================================================================
def _kombi_by_seasonpass(pendler_kombi, userdict):

    user_seasons = set()
    for cardnum, season in userdict.items():
        tups = {str((cardnum, seasonid)).encode('utf-8') for seasonid in season}
        user_seasons.update(tups)

    valid = set()
    with lmdb.open(pendler_kombi) as env:
        with env.begin() as txn:
            for card in user_seasons:
                try:
                    v = txn.get(card)
                except KeyError:
                    continue
                if v:
                    v = v.decode('utf-8')
                    v = set(ast.literal_eval(v))
                    valid.update(v)
    return valid

def _get_trips(db_path, tripkeys, model):

    tripkeys_ = (bytes(str(x), 'utf-8') for x in tripkeys)
    errors = {'operator_error', 'station_map_error', 'rk_operator_error', 'no_available_trip'}

    out = {}
    with lmdb.open(db_path) as env:
        with env.begin() as txn:
            for k in tripkeys_:
                res = txn.get(k)
                if not res:
                    continue
                res = res.decode('utf-8')
                if res not in errors:
                    val = ast.literal_eval(res)
                    # if model != 3:
                    out[int(k.decode('utf-8'))] = val
                    # else:
                        # out[int(k.decode('utf-8'))] = pendler_reshare(val)

    return out


def _npaid_zones(userdict, valid_kombi_store, zero_travel_price, db_path, year, model):


    takstsets = ["vestsjælland", "sydsjælland", "hovedstad", "dsb"]

    for takst in takstsets:
        out = {}
        for nzones in tqdm(range(1, 100), f'calculating kombi paid zones - {takst}'):
            if nzones == 97:
                paidzones = None
            else:
                paidzones = nzones
            if nzones == 99:
                out[nzones] = out[97]
                break
            all_users = userdict.get_data(paid_zones=paidzones, takst=takst)
            usertrips = _kombi_by_seasonpass(valid_kombi_store, all_users)
            trips = usertrips.intersection(zero_travel_price)
            tripshares = _get_trips(db_path, trips, model)
            shared = get_user_shares(tripshares.values())
            shared['n_users'] = len(all_users)
            shared['n_period_cards'] = sum(len(x) for x in all_users.values())
            out[nzones] = shared

        frame = pd.DataFrame.from_dict(out, orient='index')
        frame.index.name = 'betaltezoner'
        frame = frame.reset_index()
        frame = frame.fillna(0)
        colorder = [x for x in frame.columns if x not in ('n_users', 'n_period_cards', 'n_trips')]
        colorder = colorder + ['n_users', 'n_period_cards', 'n_trips']
        frame = frame[colorder]

        fp = os.path.join(
            THIS_DIR,
            '__result_cache__',
            f'{year}',
            'pendler',
            f'kombi_paid_zones_region_{takst}_model_{model}.csv'

            )
        frame.to_csv(fp, index=False)


def _chosen_zones(
        userdict,
        db_path,
        kombi_valid_db,
        kombi_dates_db,
        zero_travel_price,
        year,
        model
        ):

    userdata = userdict.get_data()
    zone_combinations = get_zone_combinations(userdata)

    zone_combo_users, statistics = \
        get_users_for_zones(userdict, zone_combinations)

    season_times = proc_user_data(userdata, zone_combo_users)

    kombi_trips = load_valid(kombi_valid_db)

    zone_combo_trips = trips_to_zone_combination(
        kombi_trips, zone_combo_users
        )

    zone_combo_trips_valid = assert_internal_zones(
        zero_travel_price, zone_combo_trips
        )

    results = get_zone_combination_shares(
        zone_combo_trips_valid, db_path, model
        )

    results = {tuple(sorted(k)): v for k, v in results.items()}
    statistics = {tuple(sorted(k)):v for k, v in statistics.items()}
    for k, v in results.copy().items():
        results[k]['n_users'] = statistics[k]['n_users']
        results[k]['n_period_cards'] = statistics[k]['n_period_cards']
    results = {str(k): v for k, v in results.items()}

    out = pd.DataFrame.from_dict(results, orient='index')
    out = out.fillna(0)
    colorder = [
        x for x in out.columns if x
        not in ('n_users', 'n_period_cards', 'n_trips')
        ]
    colorder = colorder + ['n_users', 'n_period_cards', 'n_trips']
    out = out[colorder]

    fp = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}',
        'pendler',
        f'pendlerchosenzones{year}_model_{model}.csv'
        )
    out.to_csv(fp)

# =============================================================================
# match the zone_relations
# =============================================================================
def _load_zone_relations():

    fp = pkg_resources.resource_filename(
        'tablesalt',
        '/resources/revenue/zone_relations.msgpack'
        )

    with open(fp, 'rb') as f:
        zone_relations = msgpack.load(f, strict_map_key=False)
    return zone_relations

def _unpack_valid_zones(zonelist):

    return tuple(x['ZoneID'] for x in zonelist)

def _proc_zone_relations(zone_rels: Dict):

    wanted_keys = ('StartZone',
                   'DestinationZone',
                   'PaidZones',
                   'ValidityZones',
                   'Zones')

    zone_rels = {k: {k1: v1 for k1, v1 in v.items() if k1 in wanted_keys}
                 for k, v in zone_rels.items()}
    zone_rels = list(zone_rels.values())

    for x in zone_rels:
        x['ValidZones'] = _unpack_valid_zones(x['Zones'])

    return zone_rels

def _load_kombi_results(year: str, model: int):

    fp = (THIS_DIR / '__result_cache__' /  f'{year}' /
          'pendler' / f'pendlerchosenzones{year}_model_{model}.csv')


    results = pd.read_csv(fp, index_col=0)
    results = results.to_dict(orient='index')

    return {ast.literal_eval(k): v for k, v in results.items()}


def _zonerelations(year: int, model: int):

    zone_rels = _load_zone_relations()
    zone_rels = _proc_zone_relations(zone_rels)
    results = _load_kombi_results(year, model)

    out = {}
    for i, x in enumerate(zone_rels):
        x = {k: v for k, v in x.items() if k != 'Zones'}
        matched_result = results.get(x['ValidZones'])
        if matched_result is not None:
            val = {**x, **matched_result}
        else:
            val = x
        out[i] = val

    frame = pd.DataFrame.from_dict(out, orient='index')
    cp = frame.copy(deep=True)

    cp.rename(columns={'StartZone': 'DestinationZone', 'DestinationZone': 'StartZone'}, inplace=True)
    df = pd.concat([frame, cp])
    df = df.fillna(0)
    # add new frame d=o and o=d
    fp = (THIS_DIR / '__result_cache__' / f'{year}' / 'pendler' /
          f'zonerelations{year}_model_{model}.csv')

    df.to_csv(fp, index=False)


def main():

    parser = TableArgParser(
        'year', 'products', 'zones',
        )

    args = parser.parse()

    year = args['year']
    paths = db_paths(find_datastores(), year)
    stores = paths['store_paths']
    db_path = paths['calculated_stores']
    zone_path = args['zones']
    product_path = args['products']
    zero_travel_price = find_no_pay(stores, year)

    userdict = PendlerKombiUsers(
        year,
        products_path=product_path,
        product_zones_path=zone_path,
        min_valid_days=14
        )

    for model in [1, 2, 3, 4]:
        if model > 1:
            result_path = db_path + f'_model_{model}'
        else:
            result_path = db_path


        _chosen_zones(
            userdict,
            result_path,
            paths['kombi_valid_trips'],
            paths['kombi_dates_db'],
            zero_travel_price,
            year,
            model
            )

        _npaid_zones(
            userdict,
            paths['kombi_valid_trips'],
            zero_travel_price,
            result_path,
            year,
            model
            )

        _zonerelations(year, model)

if __name__ == "__main__":
    st = datetime.now()

    INHIBITOR = None
    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
    main()

    if INHIBITOR:
        INHIBITOR.uninhibit()

    print(datetime.now() - st)