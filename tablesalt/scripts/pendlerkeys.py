# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:31:46 2020

@author: alkj
"""
import ast
import os
from datetime import datetime
from itertools import groupby, chain
from multiprocessing import Pool
from operator import itemgetter

import lmdb
import pandas as pd
from tqdm import tqdm

from tablesalt import StoreReader
from tablesalt.common.io import mappers
from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser



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

    return zone_set

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
        for res in tqdm(results, 'finding trips inside zones'):
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
    env = lmdb.open(kombi_dates)
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
    env.close()
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
    rev = {v:k for k, v in mappers['operator_id'].items()}
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

    env = lmdb.open(calculated_stores)

    final = {}
    with env.begin() as txn:
        for k, v in tqdm(tofetch.items(), 'fetching valid trip results'):
            all_trips = []
            for trip in v:
                t = txn.get(trip.encode('utf-8'))
                if t:
                    all_trips.append(t.decode('utf-8'))
            all_trips = tuple(ast.literal_eval(x) for x in all_trips)
            final[k] = get_user_shares(all_trips)
    env.close()

    return final


def main():
    """
    Entry point

    Returns
    -------
    None.

    """
    parser = TableArgParser('year', 'products', 'zones')

    args = parser.parse()

    paths = db_paths(find_datastores('H:/'), args['year'])
    stores = paths['store_paths']

    year = args['year']
    zone_path = args['zones']
    product_path = args['products']

    userdict = PendlerKombiUsers(
        year, products_path=product_path,
        product_zones_path=zone_path,
        min_valid_days=14
        )

    userdata = userdict.get_data()

    zone_combinations = get_zone_combinations(userdata)

    zone_combinations = {x for x in zone_combinations
                          if all(y < 1300 for y in x)}

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

    zero_travel_price = find_no_pay(stores)
    zero_travel_price = {str(x) for x in zero_travel_price}

    zone_combo_trips_valid = assert_internal_zones(
        zero_travel_price, zone_combo_trips
        )

    t = get_zone_combination_shares(zone_combo_trips_valid, paths['calculated_stores'])
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
        '__result_cache__', f'{year}', 
        'pendler', 'pendlerchosenzones.csv'
        )
    out.to_csv(fp, index=True)


if __name__ == "__main__":
    st = datetime.now()
    main()
    print(datetime.now() - st)
