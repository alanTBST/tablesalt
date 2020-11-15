# -*- coding: utf-8 -*-
"""
TBST - Trafik, Bygge, og Bolig -styrelsen


@author: Alan Jones
@email: alkj@tbst.dk; alanksjones@gmail.com

"""

import ast
import os
from pathlib import Path
from operator import itemgetter
from itertools import groupby, chain
from collections import defaultdict
from datetime import datetime

import lmdb
import numpy as np
from tqdm import tqdm
import pandas as pd

from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser
from pendlerkeys import find_no_pay

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

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

def _make_date(x):
    x = x.decode('utf-8')
    return datetime(*(int(y.lstrip('0')) for y in x.split('-'))).date()
# datetime.strptime(v.decode('utf-8'),'%Y-%m-%d').date()
#pd.to_datetime('2019-01-01').date()
# def _make_date(x):
#     x = x.decode('utf-8')
#     return pd.to_datetime(x).date()


def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)

def match_trip_to_season(kombi_trips, userdata, kombi_dates_store):

    out = {}
    with lmdb.open(kombi_dates_store) as env:
        with env.begin() as txn:
            for k, v in tqdm(kombi_trips.items()):
                v = (bytes(str(x), 'utf-8') for x in v)             
                utripdates = ((x, txn.get(x)) for x in v)
                
                utripdates = {
                    k.decode('utf-8'): _make_date(v)
                    for k, v in utripdates
                    }
                user_seasons = tuple(
                    (v, k1) for k1, v in userdata.items() if k1[0] == k
                    )

                for window, season in user_seasons:
                    if season not in out:
                        out[season] = []
                
                for key, date in utripdates.items():
                    for window, season in user_seasons:
                        if _date_in_window(window, date):
                            break
                    else:
                        continue
                    
                    cur = out.get(season)
                    if cur is not None:
                        cur.append(key)
                    out[season] = cur
    
    return out

def make_output(usershares, product_path, zone_path, model, year):
  
    period_products = pd.read_csv(
        product_path, sep=';', encoding='iso-8859-1'
        )
    
    period_products.Pris = \
    period_products.Pris.str.replace(',','').str.replace('.','').astype(float) / 100

    ushares = pd.DataFrame.from_dict(
        usershares, orient='index'
        )
    ushares = ushares.reset_index()
    ushares.rename(
        columns={'level_0':'EncryptedCardEngravedID',
                 'level_1':'SeasonPassID'}, 
        inplace=True
        )
    ushares = ushares.fillna(0)

    kombi = period_products.loc[
        period_products.loc[:, 'SeasonPassName'].str.lower().str.contains('kombi')
        ]
    
    pendler = period_products.loc[~
        period_products.loc[:, 'SeasonPassName'].str.lower().str.contains('kombi')
        ]

    merge1 = pd.merge(
        kombi, ushares, 
        on=['EncryptedCardEngravedID', 'SeasonPassID'], 
        how='left'
        )
    
    merge1 = merge1.fillna(0)
        
    missed = merge1.query("n_trips == 0").copy()
    kombi_match = merge1.query("n_trips != 0").copy()
    kombi_match['note'] = 'user_period_shares'
           
    missed['key'] = missed.loc[
        :, ('EncryptedCardEngravedID', 'SeasonPassID')
        ].apply(tuple, axis=1)

    # load zones
    pendler_product_zones = pd.read_csv(
        zone_path, sep=';', encoding='iso-8859-1'
        )

    pendler_product_zones = pendler_product_zones.sort_values(
        ['EncryptedCardEngravedID', 'SeasonPassID']
        )
    
    pendler_product_zones = \
    pendler_product_zones.itertuples(name=None, index=False)
    
    pendler_product_zones = {
        key: tuple(x[2] for x in group) for key, group in
        groupby(pendler_product_zones, key=itemgetter(0, 1))
        }
    
    
    missed.loc[:, 'ValidZones'] = missed['key'].map(pendler_product_zones)
    missed.loc[:, 'ValidZones'] = missed.loc[:, 'ValidZones'].astype(str)
    
    missed = missed.drop(
        ['movia','stog', 'dsb', 'first', 'metro', 'n_trips'], axis=1
        )  
    # load the chosenzones results
    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', 
        f'{year}', 
        'pendler', 
        f'pendlerchosenzones{year}_model_{model}.csv'
        )
    pendlerchosen = pd.read_csv(fp ,index_col=0)
    pendlerchosen.index.name = 'ValidZones'
    pendlerchosen = pendlerchosen.reset_index()
        
    merge2 = pd.merge(missed, pendlerchosen, on='ValidZones', how='left')
    merge2 = merge2.drop(['key','ValidZones'], axis=1)
    merge2 = merge2.fillna(0)
    
    matched = merge2.query("n_trips != 0").copy()
    matched['note'] = 'aggregate_kombi_match'
   
    missed = merge2.query("n_trips == 0").copy()
    missed['note'] = 'waiting'
    out = pd.concat([kombi_match, matched, missed])
    out = out.fillna(0)
    
    # write kombi output
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', 
        f'{year}', 
        'pendler', 
        f'kombiusershares{year}_model_{model}.csv'
        )
    out.to_csv(fp, index=False)

    pendler['key'] = pendler.loc[
        :, ('EncryptedCardEngravedID', 'SeasonPassID')
        ].apply(tuple, axis=1)

    pendler.loc[:, 'ValidZones'] = pendler['key'].map(pendler_product_zones)
    pendler.loc[:, 'ValidZones'] = pendler.loc[:, 'ValidZones'].astype(str)
    
    merge1 = pd.merge(pendler, pendlerchosen, on='ValidZones', how='left')
    merge1.fillna(0)
    merge1 = merge1.drop(['key','ValidZones'], axis=1)    
    missed = merge1.query("n_trips == 0").copy()
    
    pendler_match = merge1.query("n_trips != 0").copy()
    pendler_match['note'] = 'kombi_match'
    
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', 
        f'{year}', 
        'pendler', 
        f'zonerelations{year}_model_{model}.csv'
        )
     
    relations = pd.read_csv(fp)
    missed = missed[
        ['EncryptedCardEngravedID', 'SeasonPassID', 
         'SeasonPassTemplateID', 'SeasonPassName', 
         'Fareset', 'PsedoFareset', 'SeasonPassType',
         'PassengerGroupType1', 'SeasonPassStatus', 
         'ValidityStartDT', 'ValidityEndDT', 'ValidDays', 
         'FromZoneNr', 'ToZoneNr', 'ViaZoneNr',
         'SeasonPassZones', 'PassagerType', 
         'TpurseRequired', 'SeasonPassCategory', 
         'Pris', 'RefundType', 'productdate']
        ]
    
    relations = relations[['StartZone', 'DestinationZone', 
        'movia', 'stog', 'first', 'metro', 'dsb', 'n_users',
        'n_period_cards', 'n_trips']]  
    relations.rename(columns={
        'StartZone': 'FromZoneNr', 
        'DestinationZone': 'ToZoneNr'}, inplace=True)

    merge = pd.merge(missed, relations, on=['FromZoneNr', 'ToZoneNr'], how='left')
    merge.loc[:, 'note'] = 'from_to'
    
    out = pd.concat([pendler_match, merge])  
    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', 
        f'{year}', 
        'pendler', 
        f'rejsekort_pendler{year}_model_{model}.csv'
        )
    out.to_csv(fp, index=False)

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

def n_operators(share_tuple):
    
    return len({x[1] for x in share_tuple})


def pendler_reshare(share_tuple):
    
    if not isinstance(share_tuple[0], tuple):
        share_tuple = share_tuple, 
    n_ops = n_operators(share_tuple)
    
    return tuple((1/n_ops, x[1]) for x in share_tuple)


def fetch_trip_results(db_path, tofetch, model):
 
    with lmdb.open(db_path) as env:
        final = {}
        with env.begin() as txn:
            for card_season, trips in tqdm(tofetch.items()):
                all_trips = []
                for trip in trips:
                    t = txn.get(bytes(trip, 'utf-8'))
                    if t:
                        all_trips.append(t.decode('utf-8'))                 
                all_trips = tuple(
                    ast.literal_eval(x) for x in all_trips 
                    if x not in ('station_map_error', 'operator_error')
                    )
                if model == 3:
                    all_trips = tuple(pendler_reshare(x) for x in all_trips)                    
                final[card_season] = get_user_shares(all_trips)
    
    return final

def main():
 
    # Use zero price
    parser = TableArgParser('year', 'zones', 'products', 'model')
    args = parser.parse()

    year = args['year']
    model = args['model']
    paths = db_paths(find_datastores(), year)
    db_path = paths['calculated_stores']
    
    stores = paths['store_paths']
    
    if model == 2:
        db_path = db_path + '_model_2'

    valid_kombi_store = paths['kombi_valid_trips']
    kombi_dates =  paths['kombi_dates_db']

    userdata = PendlerKombiUsers(
        year, 
        products_path=args['products'],
        product_zones_path=args['zones'],
        min_valid_days=0
        ).get_data()

    userdata = process_user_data(userdata)
    kombi_trips = load_valid(valid_kombi_store)
    zero_travel_price = find_no_pay(stores, year)

    valid = {
        k: set(v).intersection(zero_travel_price) for 
        k, v in kombi_trips.items()
        }
    
    tofetch = match_trip_to_season(
        valid, userdata, kombi_dates
        )
    
    results = fetch_trip_results(db_path, tofetch, model)

    make_output(results, args['products'], args['zones'], model, year)
# =============================================================================
#     
# =============================================================================
if __name__ == "__main__":
    main()    
    
    
    
 