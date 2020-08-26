# -*- coding: utf-8 -*-
"""
TBST Trafik, Bygge, og Bolig -styrelsen


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com


calculate the keys for KOMBI users where
the only pendler sales information known is the total number
of zones
"""

import ast
import pickle
import sqlite3
from itertools import groupby, chain
from operator import itemgetter

import lmdb
import pandas as pd
from tqdm import tqdm

from tablesalt.season.users import UserDict
from tablesalt.common.io import mappers


YEAR = 2019

#default
share_db = r'H:\datastores\rejsekortstores\2019DataStores\dbs\calculated_stores'
# rabat_zero = r'..working\rabat0trips.pickle'
pendler_kombi = r'H:\datastores\rejsekortstores\2019DataStores\dbs\kombi_valid_trips'


PRODUCTS = r'H:\revenue\inputdata\2019\PeriodeProdukt.csv'
PRODUCT_ZONES = r'H:\revenue\inputdata\2019\Zoner.csv'



users = UserDict(
        YEAR, products_path=PRODUCTS,
        product_zones_path=PRODUCT_ZONES,
        min_valid_days=14
        )

def _kombi_by_users(cardnums):

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


def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values
    """

    test_out = sorted(shares, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)
    t = {key:sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    t = {k:v/totalzones for k, v in t.items()}

    return t

def get_shares(all_trips):

    n_trips = len(all_trips)
    single = [x for x in all_trips if isinstance(x[0], int)]
    multi =  list(chain(*[x for x in all_trips if isinstance(x[0], tuple)]))
    all_trips = single + multi
    user_shares = _aggregate_zones(all_trips)
    user_shares['n_trips'] = n_trips

    return user_shares

def _get_trips(tripkeys):


    tripkeys_ = {bytes(str(x), 'utf-8') for x in tripkeys}

    out = {}
    with lmdb.open(share_db) as env:
        with env.begin() as txn:
            for k in tripkeys_:
                shares = txn.get(k)
                if shares:
                    shares = shares.decode('utf-8')
                    out[int(k.decode('utf-8'))] = ast.literal_eval(shares)

    return out


def _all_zone_shares(ptype=None):

    all_users = users.get_data(ptype=ptype)
    usertrips = _kombi_by_users(all_users)

    tripshares = _get_trips(usertrips)
    shared = get_shares(tripshares.values())
    shared['n_users'] = len(all_users)
    shared['n_period_cards'] = sum(len(x) for x in all_users.values())

    return shared

def main():

    out = {}
    passengertype = None
    for nzones in tqdm(range(1, 100), 'calculating kombi paid zones'):
        if nzones == 99:
            out[nzones] = _all_zone_shares(ptype=passengertype)
            continue
        nzones_users = users.get_data(paid_zones=nzones)
        if not nzones_users:
            continue
        nzones_users = users.get_data(paid_zones=nzones, ptype=passengertype)

        nzones_usertrips = _kombi_by_users(nzones_users)
        tripshares = _get_trips(nzones_usertrips)
        shared = get_shares(tripshares)
        shared['n_users'] = len(nzones_users)
        shared['n_period_cards'] = sum(len(x) for x in nzones_users.values())
        out[nzones] = shared

    frame = pd.DataFrame.from_dict(out, orient='index')
    frame.to_csv(f'kombi_all_zones_{passengertype}.csv')


if __name__ == "__main__":
    main()