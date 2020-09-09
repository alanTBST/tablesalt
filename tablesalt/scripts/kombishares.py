# -*- coding: utf-8 -*-
"""
TBST - Trafik, Bygge, og Bolig -styrelsen


@author: Alan Jones
@email: alkj@tbst.dk; alanksjones@gmail.com

"""

import ast
import glob
import os
import sys
from pathlib import Path
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )


from operator import itemgetter
from itertools import groupby, chain
from collections import defaultdict
from datetime import datetime

import lmdb
import numpy as np
import tqdm
import pandas as pd

from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser


def parse_args():
    """parse the cl arguments"""
    DESC = ("Setup all the key-value stores needed \n"
            "for the pendler card revenue distribution \n"
            "for takstsjælland.")

    parser = ArgumentParser(
        description=DESC,
        formatter_class=RawTextHelpFormatter
        )
    parser.add_argument(
        '-y', '--year',
        help='year to unpack',
        type=int,
        required=True
        )
    parser.add_argument(
        '-z', '--zones',
        help='path to input zones csv',
        type=Path,
        required=True
        )
    parser.add_argument(
        '-p', '--products',
        help='path to input pendler products csv',
        type=Path,
        required=True
        )

    args = parser.parse_args()

    return vars(args)


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


def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values
    """

    test_out = sorted(shares, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)
    # rev = {v:k for k, v in mappers['operator_id'].items()}
    t = {key:sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    t = {k:v/totalzones for k, v in t.items()}

    return t


def load_valid(valid_kombi_store):
    """
    Load the valid kombi user trips

    Returns
    -------
    valid_kombi : dict
        DESCRIPTION.

    """
    env = lmdb.open(valid_kombi_store)
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

def match_trip_to_season(kombi_trips, userdata, kombi_dates_store):

    out = {}
    env = lmdb.open(kombi_dates_store)
    with env.begin() as txn:
        i = 0
        for k, v in kombi_trips.items():
            v = (bytes(str(x), 'utf-8') for x in v)
            utripdates = {x: txn.get(x) for x in v}
            utripdates = {
                k.decode('utf-8'): datetime.strptime(v.decode('utf-8'),'%Y-%m-%d').date()
                for k, v in utripdates.items()
                }
            user_seasons = {v: k1 for k1, v in userdata.items() if k1[0] == k}
            for window, season in user_seasons.items():
                if season not in out:
                    out[season] = []
            for key, date in utripdates.items():
                for window, season in user_seasons.items():
                    if _date_in_window(window, date):
                        break
                cur = out.get(season)
                if cur is not None:
                    cur.append(key)
                out[season] = cur
            if i % 100 == 0:
                print(i)
            i+=1
    env.close()

    return out


def make_output(usershares, product_path):

    cardnum = {x[0] for x in usershares}
    prods = pd.read_csv(product_path, sep=';', encoding='iso-8859-1')
    prods = prods.query("EncryptedCardEngravedID in @cardnum")


    ushares = pd.DataFrame.from_dict(usershares, orient='index')
    ushares = ushares.reset_index()
    ushares.rename(columns={'level_0':'EncryptedCardEngravedID',
                            'level_1':'SeasonPassID'}, inplace=True)
    ushares = ushares.fillna(0)
    out = pd.merge(prods, ushares, on=['EncryptedCardEngravedID', 'SeasonPassID'])

    return out

def divide_dict(dictionary, chunk_size):

    """
    Divide one dictionary into several dictionaries
    Return a list, each item is a dictionary
    """
    count_ar = np.linspace(0, len(dictionary), chunk_size+1, dtype= int)
    group_lst = []
    temp_dict = defaultdict(lambda : None)
    i = 1
    for key, value in dictionary.items():
        temp_dict[key] = value
        if i in count_ar:
            group_lst.append(temp_dict)
            temp_dict = defaultdict(lambda : None)
        i += 1
    return [dict(x) for x in group_lst]


def process_user_data(udata):

    outdict = {}
    for k, v in udata.items():
        for k1, v1 in v.items():
            outdict[(k, k1)] = v1['start'].date(), v1['end'].date()

    return outdict

def get_user_shares(all_trips):

    n_trips = len(all_trips)
    single = [x for x in all_trips if isinstance(x[0], int)]
    multi =  list(chain(*[x for x in all_trips if isinstance(x[0], tuple)]))
    all_trips = single + multi
    user_shares = _aggregate_zones(all_trips)
    user_shares['n_trips'] = n_trips

    return user_shares

def main():

    parser = TableArgParser('year', 'zones', 'products')
    args = parser.parse()

    year = args['year']
    zone_path = args['zones']
    product_path = args['products']

    paths = db_paths(find_datastores('H:/'), year)
    calc_store = paths['calculated_stores']
    valid_kombi_store = paths['kombi_valid_trips']
    kombi_dates =  paths['kombi_dates_db']

    userdata = PendlerKombiUsers(
        year, products_path=product_path,
        product_zones_path=zone_path,
        min_valid_days=0
        ).get_data()

    userdata = process_user_data(userdata)
    kombi_trips = load_valid(valid_kombi_store)
    tofetch = match_trip_to_season(
        kombi_trips, userdata, kombi_dates
        )

    with lmdb.open(calc_store) as env:
        final = {}
        with env.begin() as txn:
            for k, v in tofetch.items():
                all_trips = []
                for trip in v:
                    t = txn.get(trip.encode('utf-8'))
                    if t:
                        all_trips.append(t.decode('utf-8'))
                all_trips = tuple(ast.literal_eval(x) for x in all_trips)
                final[k] = get_user_shares(all_trips)

    out = make_output(final, product_path)
    cols = [x for x in out.columns if x != 'n_trips']
    colorder = cols + ['n_trips']
    out = out[colorder]
    
    fp = os.path.join(
        '__result_cache__', f'{year}', 'pendler', 'kombiusershares.csv'
        )
    out.to_csv(fp, index=False)
