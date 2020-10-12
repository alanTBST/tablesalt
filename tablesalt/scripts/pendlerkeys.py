# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:31:46 2020

@author: alkj
"""
import ast
import os
import pkg_resources
from datetime import datetime
from itertools import groupby, chain
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path

import lmdb
import msgpack
import pandas as pd
from tqdm import tqdm

from tablesalt import StoreReader
from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

def get_zone_combinations(udata):
    """
    Get all the chosen zone combinations of pendler kombi users

    Parameters
    ----------
    udata : dict
        userdata loaded from users.PendlerInput class.

    Returns
    -------
    zone_set : set
        distinct valid zone combinations of all users.

    """

    zone_set = set()
    for _, seasons in udata.items():
        for _id, card_info in seasons.items():
            if card_info['zones'] not in zone_set:
                zone_set.add(card_info['zones'])
    # THIS IS ONLY FOR SJÆLLAND
    return {x for x in zone_set if all(y < 1300 for y in x)}

def get_users_for_zones(udict, zone_set):
    """
    For each distinct zone combination, find their users and season passes

    Parameters
    ----------
    udict : users.PendlerKombiUsers
        the user dictionary for pendler kombi users.
    zone_set : set
        distinct valid zone combinations of all users .

    Returns
    -------
    zone_set_users : dict
        DESCRIPTION.
    statistics : dict
        DESCRIPTION.

    """


    zone_set_users = {}
    statistics = {}

    for zone_combo in zone_set:
        combo_users = udict.subset_zones(zones=zone_combo)
        card_seasons = {k: tuple(v.keys()) for
                        k, v in combo_users[0].items()}
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



def load_valid(valid_kombi_store):
    """
    Load the valid kombi user trips

    Returns
    -------
    valid_kombi : dict
        DESCRIPTION.

    """
    env = lmdb.open(valid_kombi_store, readahead=False)
    valid_kombi = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            valid_kombi[k.decode('utf-8')] = ast.literal_eval(v.decode('utf-8'))
    env.close()
    return valid_kombi

def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)

def wrap_price(store):
    # TODO: rather use partial
    return StoreReader(store).get_data('price')

def get_zeros(p):
    df = pd.DataFrame(p[:, (0, -1)],
                      columns=['tripkey', 'price']).set_index('tripkey')
    df = df.groupby(level=0)['price'].transform(max)
    df = df[df == 0]
    return tuple(set(df.index.values))

def proc(store):

    price = wrap_price(store)

    return get_zeros(price)

def find_no_pay(stores):

    out = set()
    with Pool(os.cpu_count() - 2) as pool:
        results = pool.imap(proc, stores)
        for res in tqdm(results, 
                        'finding trips inside zones', 
                        total=len(stores)):
            out.update(set(res))
    return out

def assert_internal_zones(zero_travel_price, zone_combo_trips):

    return {k: v.intersection(zero_travel_price) for k, v in zone_combo_trips.items()}

def match_trip_to_season(kombi_trips, season_dates, kombi_dates):
    """
    For each user seasonpassID match the valid kombi trips to
    that seasonpassID time window

    Parameters
    ----------
    kombi_trips : dict
        DESCRIPTION.
    season_dates : dict
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    out = {k: [] for k, v in season_dates.items()}
    with lmdb.open(kombi_dates, readahead=False) as env:
        with env.begin() as txn:
            for k, v in tqdm(kombi_trips.items(), 'matching trips to season passes'):
                v = (bytes(str(x), 'utf-8') for x in v)
                utripdates = {x: txn.get(x) for x in v}
                utripdates = {
                    k.decode('utf-8'): datetime.strptime(
                        v.decode('utf-8'), '%Y-%m-%d'
                        ).date() for k, v in utripdates.items()
                    }
                user_seasons = {v: k1 for k1, v in
                                season_dates.items() if k1[0] == k}
                for key, date in utripdates.items():
                    for window, season in user_seasons.items():
                        if _date_in_window(window, date):
                            out[season].append(key)
                            break
    return {k: tuple(v) for k, v in out.items()}


def trips_to_zone_combination(season_trips, zone_combo_users):

    zone_combo_trips = {}
    for zones, userseasons in zone_combo_users.items():
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

def get_zone_combination_shares(tofetch, calculated_stores):

    with lmdb.open(calculated_stores) as env:
        final = {}
        with env.begin() as txn:
            for combo, trips in tofetch.items():
                all_trips = {}
                for trip in trips:
                    t = bytes(trip, 'utf-8')
                    res = txn.get(t)
                    if not res:
                        continue
                    res = res.decode('utf-8')
                    
                    if res not in ('operator_error', 'station_map_error'):
                        all_trips[trip] = ast.literal_eval(res)
                combo_result = get_user_shares(all_trips.values())
                final[combo] = combo_result

    return final

# =============================================================================
# aggregation by paid zones
# =============================================================================
def _kombi_by_users(pendler_kombi, cardnums):

    cardnums = {x.encode('utf-8') for x in cardnums}
    valid = set()
    with lmdb.open(pendler_kombi) as env:
        with env.begin() as txn:
            for card in cardnums:
                try:
                    v = txn.get(card)
                except KeyError:
                    continue
                if v:
                    v = v.decode('utf-8')
                    v = set(ast.literal_eval(v))
                    valid.update(v)
    return valid

def _get_trips(share_db, tripkeys):


    tripkeys_ = {bytes(str(x), 'utf-8') for x in tripkeys}

    out = {}
    with lmdb.open(share_db) as env:
        with env.begin() as txn:
            for k in tripkeys_:
                res = txn.get(k)
                if not res:
                    continue
                res = res.decode('utf-8')
                    
                if res not in ('operator_error', 'station_map_error'):
                    out[int(k.decode('utf-8'))] = ast.literal_eval(res)
    return out

def _npaid_zones(userdict, valid_kombi_store, zero_travel_price, calc_store, year):
    
    zero_travel_price = {int(x) for x in zero_travel_price}
    takstsets = ["vestsjælland", "sydsjælland", "hovedstad", "dsb"]
    
    frames = []
    for takst in takstsets:
        out = {}
        for nzones in tqdm(range(1, 100), f'calculating kombi paid zones - {takst}'):
            if nzones in (99, 97):
                paidzones = None
            else:
                paidzones = nzones
            all_users = userdict.get_data(paid_zones=paidzones, takst=takst)
            usertrips = _kombi_by_users(valid_kombi_store, all_users)
            trips = usertrips.intersection(zero_travel_price)
            tripshares = _get_trips(calc_store, trips)
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
            f'kombi_paid_zones_region_{takst}.csv'
            )
        frame.to_csv(fp, index=False)


# aggregated by chosenzones
def _chosen_zones(userdict, paths, zero_travel_price, year):
    
    
    userdata = userdict.get_data()
    zone_combinations = get_zone_combinations(userdata)

    zone_combo_users, statistics = \
        get_users_for_zones(userdict, zone_combinations)

    season_times = proc_user_data(userdata, zone_combo_users)

    kombi_trips = load_valid(paths['kombi_valid_trips'])

    season_trips = match_trip_to_season(
        kombi_trips, season_times, paths['kombi_dates_db']
        )

    zone_combo_trips = trips_to_zone_combination(
        season_trips, zone_combo_users
        )


    zone_combo_trips_valid = assert_internal_zones(
        zero_travel_price, zone_combo_trips
        )

    t = get_zone_combination_shares(
        zone_combo_trips_valid, paths['calculated_stores']
        )
       
    t = {tuple(sorted(k)):v for k, v in t.items()}
    statistics = {tuple(sorted(k)):v for k, v in statistics.items()}
    for k, v in t.copy().items():
        t[k]['n_users'] = statistics[k]['n_users']
        t[k]['n_period_cards'] = statistics[k]['n_period_cards']
    t = {str(k): v for k, v in t.items()}

    out = pd.DataFrame.from_dict(t, orient='index')
    out = out.fillna(0)
    colorder = [x for x in out.columns if x not in ('n_users', 'n_period_cards', 'n_trips')]
    colorder = colorder + ['n_users', 'n_period_cards', 'n_trips']
    out = out[colorder]
    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', f'{year}', 
        'pendler', 'pendlerchosenzones.csv'
        )
    out.to_csv(fp)
    
    return out

# =============================================================================
# match the zone_relations
# =============================================================================
def _load_zone_relations():
    """
    Returns
    -------
    zonerelations : TYPE
        DESCRIPTION.

    """
    
    fp = pkg_resources.resource_filename(
        'tablesalt', 
        '/resources/revenue/zone_relations.msgpack'
        )
        
    with open(fp, 'rb') as f:
        zone_relations = msgpack.load(f, strict_map_key=False)


    return zone_relations

def _unpack_valid_zones(zonelist):

    return tuple(x['ZoneID'] for x in zonelist)

def _proc_zone_relations(zone_rels: dict):

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

def _load_kombi_results(year):

    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', f'{year}', 
        'pendler', 'pendlerchosenzones.csv'
        )
    
    results = pd.read_csv(fp, index_col=0)
    results = results.to_dict(orient='index')

    return {ast.literal_eval(k): v for k, v in results.items()}

    
def _zonerelations(year):

    zone_rels = _load_zone_relations()
    zone_rels = _proc_zone_relations(zone_rels)
    results = _load_kombi_results(year)

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
    cp.columns = ['DestinationZone', 'StartZone', 'PaidZones', 'ValidityZones',
       'ValidZones', 'first', 'metro', 'movia', 'stog', 'dsb', 'n_users',
       'n_period_cards', 'n_trips']
    df = pd.concat([frame, cp])
    df = df.fillna(0)
    
    # add new frame d=o and o=d
    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', f'{year}', 
        'pendler', 'zonerelations.csv'
        )
       
    df.to_csv(fp, index=False)


def main():

    parser = TableArgParser('year', 'products', 'zones', 'model')

    args = parser.parse()

    paths = db_paths(find_datastores(), args['year'])
    stores = paths['store_paths']
    db_path = paths['calculated_stores']
    model = args['model']
    if model != 1:
        db_path = db_path + f'_model_{model}'
        

    year = args['year']
    zone_path = args['zones']
    product_path = args['products']
        
    userdict = PendlerKombiUsers(
        year, products_path=product_path,
        product_zones_path=zone_path,
        min_valid_days=14
        )

    zero_travel_price = find_no_pay(stores)
    zero_travel_price = {str(x) for x in zero_travel_price}

    _chosen_zones(userdict, paths, zero_travel_price, year)
   
    _npaid_zones(
        userdict, paths['kombi_valid_trips'], 
        zero_travel_price, paths['calculated_stores'],
        year
        )
    
    # _zonerelations(year)


if __name__ == "__main__":
    st = datetime.now()
    main()
    print(datetime.now() - st)
