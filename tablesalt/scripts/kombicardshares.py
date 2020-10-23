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
from tqdm import tqdm
import pandas as pd

from tablesalt.season.users import PendlerKombiUsers
from tablesalt.preprocessing.tools import find_datastores, db_paths
from tablesalt.preprocessing.parsing import TableArgParser


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


def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)

def match_trip_to_season(kombi_trips, userdata, kombi_dates_store):

    out = {}
    env = lmdb.open(kombi_dates_store)
    with env.begin() as txn:
        for k, v in tqdm(kombi_trips.items()):

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

    env.close()

    return out


def make_output(usershares, product_path):

    # cardnum = {x[0] for x in usershares}
    prods = pd.read_csv(product_path, sep=';', encoding='iso-8859-1')
    # prods = prods.query("EncryptedCardEngravedID in @cardnum")


    ushares = pd.DataFrame.from_dict(usershares, orient='index')
    ushares = ushares.reset_index()
    ushares.rename(columns={'level_0':'EncryptedCardEngravedID',
                            'level_1':'SeasonPassID'}, inplace=True)
    ushares = ushares.fillna(0)

    out = pd.merge(
        prods, ushares, 
        on=['EncryptedCardEngravedID', 'SeasonPassID'], 
        how='left'
        )


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

    parser = TableArgParser('year', 'zones', 'products', 'model')
    args = parser.parse()

    year = args['year']
    model = args['model']
    zone_path = args['zones']
    product_path = args['products']
    

    paths = db_paths(find_datastores('H:/'), year)
    db_path = paths['calculated_stores']
    if model != 1:
        db_path = db_path + f'_model_{model}'

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

    with lmdb.open(db_path) as env:
        final = {}
        with env.begin() as txn:
            for k, v in tqdm(tofetch.items()):
                all_trips = []
                for trip in v:
                    t = txn.get(bytes(trip, 'utf-8'))
                    if t:
                        all_trips.append(t.decode('utf-8'))                    
                all_trips = tuple(
                    ast.literal_eval(x) for x in all_trips 
                    if x not in ('station_map_error', 'operator_error')
                    )
                final[k] = get_user_shares(all_trips)

    out = make_output(final, product_path)
    cols = [x for x in out.columns if x != 'n_trips']
    colorder = cols + ['n_trips']
    out = out[colorder]
    
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', f'{year}', 'pendler', f'kombiusershares{year}_model_{model}.csv'
        )
    out.to_csv(fp, index=False)
      
    
    period_products_fp = r'H:/revenue/inputdata/2019/PeriodeProdukt.csv'
    period_zones_fp = r'H:/revenue/inputdata/2019/Zoner.csv'
    
    
    period_products = pd.read_csv(period_products_fp, sep=';', encoding='iso-8859-1')
    pendler_product_zones = pd.read_csv(period_zones_fp, sep=';', encoding='iso-8859-1')

    pendler_product_zones['key'] = list(zip(
        pendler_product_zones['EncryptedCardEngravedID'],
        pendler_product_zones['SeasonPassID']
        ))

    pendler_product_zones = \
    pendler_product_zones.sort_values(
        ['EncryptedCardEngravedID', 'SeasonPassID']
        )
    pendler_product_zones = \
    pendler_product_zones.itertuples(name=None, index=False)
    
    pendler_product_zones = {
        key: tuple(x[2] for x in group) 
        for key, group in groupby(
                pendler_product_zones, key=itemgetter(0, 1)
                )
        }
    pendler_product_zones = [
        (k[0], k[1], str(v))
        for k, v in pendler_product_zones.items()
        ]
    pp_zones = pd.DataFrame(pendler_product_zones)
    pp_zones.columns = [
        'EncryptedCardEngravedID', 
        'SeasonPassID', 
        'ValidZones'
        ]
    
    period_products = pd.merge(
        period_products, pp_zones, 
        on=['EncryptedCardEngravedID', 'SeasonPassID'], 
        how='left'
        )
    
    pendler = period_products.loc[~
        period_products.loc[:, 'SeasonPassName'].str.lower().str.contains('kombi')
        ]

    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', f'{year}', 'pendler', f'pendlerchosenzones{year}_model_{model}.csv'
        )
    
    pendler_kombi = pd.read_csv(fp, index_col=0)
    pendler_kombi.index.name = 'ValidZones'
    pendler_kombi = pendler_kombi.reset_index()
    pendler_kombi.loc[:, 'ValidZones'] = pendler_kombi.loc[:, 'ValidZones'].astype(str)
    
    test = pd.merge(pendler, pendler_kombi, on='ValidZones', how='left')
   
    rpl = {'Sjælland': 'dsb', 
            'Vestsjælland': 'vestsjælland', 
            'Hovedstaden': 'hovedstad', 
            'Sydsjælland': 'sydsjælland'}
    
    test.loc[:, 'PsedoFareset'] = test.loc[:, 'PsedoFareset'].replace(rpl) 
    
    missed = test[test.n_trips.isnull()]    
    good = test[~test.n_trips.isnull()]
    good.loc[:, 'note'] = 'kombi_match'
    
    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', f'{year}', 'pendler', f'zonerelations{year}_model_{model}.csv'
        )
    
    
    relations = pd.read_csv(fp)

    missed = missed[['EncryptedCardEngravedID', 'SeasonPassID', 'SeasonPassTemplateID',
        'SeasonPassName', 'Fareset', 'PsedoFareset', 'SeasonPassType',
        'PassengerGroupType1', 'SeasonPassStatus', 'ValidityStartDT',
        'ValidityEndDT', 'ValidDays', 'FromZoneNr', 'ToZoneNr', 'ViaZoneNr',
        'SeasonPassZones', 'PassagerType', 'TpurseRequired',
        'SeasonPassCategory', 'Pris', 'RefundType', 'productdate', 'ValidZones']]
    
    relations = relations[['StartZone', 'DestinationZone', 
        'movia', 'stog', 'first', 'metro', 'dsb', 'n_users',
        'n_period_cards', 'n_trips']]  
    relations.rename(columns={
        'StartZone': 'FromZoneNr', 
        'DestinationZone': 'ToZoneNr'}, inplace=True)

    merge = pd.merge(missed, relations, on=['FromZoneNr', 'ToZoneNr'], how='left')
    merge.loc[:, 'note'] = 'from_to'
    
    out = pd.concat([good, merge])
    out.to_csv(f'rejsekort_pendler{year}_model_{model}.csv', index=False)
    
    
# if __name__ == "__main__":
#     main()    
    
    
    
 