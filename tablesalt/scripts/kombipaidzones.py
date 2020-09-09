# -*- coding: utf-8 -*-
"""
TBST Trafik, Bygge, og Bolig -styrelsen


Author: Alan Jones alkj@tbst.dk, alanksjones@gmail.com


calculate the keys for KOMBI users where
the only pendler sales information known is the total number
of zones
"""

import ast
import glob
import os
import sys
from itertools import groupby, chain
from multiprocessing import Pool
from operator import itemgetter


import lmdb
import pandas as pd
from tqdm import tqdm

from tablesalt import StoreReader
from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser

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

def _get_trips(share_db, tripkeys):


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

def wrap_price(store):
    # TODO: rather use partial
    return StoreReader(store).get_data('price')

def get_zeros(p):
    df = pd.DataFrame(
        p[:, (0, -1)],
        columns=['tripkey', 'price']
        ).set_index('tripkey')

    df = df.groupby(level=0)['price'].transform(max)
    df = df[df == 0]
    return tuple(set(df.index.values))

def proc(store):

    price = wrap_price(store)

    return get_zeros(price)

def find_no_pay(stores):

    out = set()
    with Pool(os.cpu_count() - 1) as pool:
        results = pool.imap(proc, stores)
        for res in tqdm(results, 'finding trips inside zones', total=len(stores)):
            out.update(set(res))
    return out

def main():

    parser = TableArgParser('year', 'products', 'zones')

    args = parser.parse()

    paths = db_paths(find_datastores('H:/'), args['year'])
    stores = paths['store_paths']
    calc_store = paths['calculated_stores']
    valid_kombi_store = paths['kombi_valid_trips']


    print("loading user data...\n")
    users = PendlerKombiUsers(
            args['year'],
            products_path=args['products'],
            product_zones_path=args['zones'],
            min_valid_days=14
            )
    print("determining valid trips...\n")
    print("\n")
    zero_travel_price = find_no_pay(stores)

    takstsets = ["vestsjælland", "sydsjælland", "hovedstad", "dsb"]

    for takst in takstsets:
        out = {}
        for nzones in tqdm(range(1, 100), f'calculating kombi paid zones - {takst}'):
            if nzones in (99, 97):
                paidzones = None
            else:
                paidzones = nzones
            all_users = users.get_data(paid_zones=paidzones, takst=takst)
            usertrips = _kombi_by_users(valid_kombi_store, all_users)
            trips = usertrips.intersection(zero_travel_price)
            tripshares = _get_trips(calc_store, trips)
            shared = get_shares(tripshares.values())
            shared['n_users'] = len(all_users)
            shared['n_period_cards'] = sum(len(x) for x in all_users.values())
            out[nzones] = shared

        frame = pd.DataFrame.from_dict(out, orient='index')
        frame.index.name = "betaltezoner"
        frame = frame.reset_index()
        frame = frame.fillna(0)
        frame.to_csv(f'sjælland/kombi_paid_zones_region_{takst}.csv', index=False)

    return

if __name__ == "__main__":
    main()