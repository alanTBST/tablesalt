# -*- coding: utf-8 -*-
"""
What does it do?
================

This script calculates the shares for the individual Rejsekort Kombi products
Each kombi card/seasonpass uses the exact valid trips made by the user and shares the price
between the operators.

For the non-kombi products, the shares are matched using the same methods as for other pendler
products
"""

import ast
import glob
import os
from datetime import datetime
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List, Tuple, Union

import lmdb
import msgpack
import pandas as pd
import pkg_resources
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import db_paths, find_datastores
from tablesalt.running import WindowsInhibitor
from tablesalt.season.users import PendlerKombiUsers
from tablesalt.topology.tools import determine_takst_region
from tqdm import tqdm

from pendlerkeys import find_no_pay

THIS_DIR = Path(__file__).parent

USER_SHARES = Dict[Tuple[str, int], Dict[str, Union[int, float]]]

TRIP_ERRORS = {
    'operator_error',
    'station_map_error',
    'rk_operator_error',
    'no_available_trip'
    }

def sort_df_by_colums(df):

    priority = {
        'EncryptedCardEngravedID': 1,
        'SeasonPassID': 2,
        'SeasonPassTemplateID': 3,
        'ProductName': 4,
        'Fareset': 5,
        'PsedoFareset': 6,
        'takstsæt': 6,
        'SeasonPassType': 7,
        'PassengerGroupType1': 8,
        'SeasonPassStatus': 9,
        'ValidityStartDT_Cal': 10,
        'ValidityEndDT_Cal': 11,
        'NumberOfPeriods_Cal': 12,
        'ZoneNrLow': 13,
        'startzone': 14,
        'ZoneNrHigh': 15,
        'slutzone': 16,
        'ViaZoneNr': 17,
        'SeasonPassZones': 18,
        'PassagerType': 19,
        'TpurseRequired': 20,
        'SeasonPassCategory': 21,
        'Price': 22,
        'RefundType': 23,
        'productdate': 24,
        'valgtezoner': 25,
        'betaltezoner': 26,
        'dsb': 27,
        'DSB': 28,
        's-tog': 29,
        'DSB S-tog': 30,
        'first': 31,
        'metro': 32,
        'Metroselskabet': 33,
        'Movia': 34,
        'movia_h': 35,
        'movia_v': 36,
        'movia_s': 37,
        'DSB_andel': 38,
        'DSB S-tog_andel': 39,
        'first_andel': 40,
        'Metroselskabet_andel': 41,
        'Movia_andel': 42,
        'movia_h_andel': 38,
        'movia_v_andel': 39,
        'movia_s_andel': 40,
        'n_trips': 41,
        'n_users': 42,
        'n_period_cards': 43,
        'note': 44
    }

    q = PriorityQueue()

    cols = df.columns
    for col in cols:
        try:
            q.put((priority[col], col))
        except KeyError:
            pass

    column_order = []
    while not q.empty():
        column_order.append(q.get()[1])

    return df[column_order]


def _aggregate_zones(shares):
    """
    aggregate the individual operator
    assignment values
    """

    test_out = sorted(shares, key=itemgetter(1))
    totalzones = sum(x[0] for x in test_out)

    t = {key: sum(x[0] for x in group) for
         key, group in groupby(test_out, key=itemgetter(1))}
    try:
        t = {k: v/totalzones for k, v in t.items()}
    except ZeroDivisionError:
        return {}
    return t


def load_valid(valid_kombi_store: str):
    """

    :param valid_kombi_store: DESCRIPTION
    :type valid_kombi_store: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    with lmdb.open(valid_kombi_store) as env:
        valid_kombi = {}
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                valid_kombi[k.decode('utf-8')] = ast.literal_eval(v.decode('utf-8'))

    return valid_kombi

def _process_for_merge(pendler_product_zones):

    pendler_product_zones = {k:str(v) for k, v in pendler_product_zones.items()}
    df = pd.DataFrame.from_dict(pendler_product_zones, orient='index')
    df = df.reset_index()
    df.columns = ['usertuple', 'valgtezoner']
    df['EncryptedCardEngravedID'], df['SeasonPassID'] = zip(*df.loc[:, 'usertuple'])

    df = df[['EncryptedCardEngravedID', 'SeasonPassID', 'valgtezoner']]
    df['valgtezoner'] = df['valgtezoner'].apply(lambda x: ast.literal_eval(x))

    return df


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


    return _proc_zone_relations(zone_relations)

def _load_process_zones(zone_path):
    # load zones
    pendler_product_zones = pd.read_csv(
        zone_path, sep=';', encoding='utf8'
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

    return _process_for_merge(pendler_product_zones)

def _get_closest_kombi(chosen_zones):

    # only for zone 1001/1002 problem right now
    new_chosen_zones = [1001] + list(chosen_zones)
    return tuple(new_chosen_zones)


def _get_closest_chosen_zones(kombi_results, chosen_zones):

    takst = determine_takst_region(chosen_zones)
    test_combination = set(chosen_zones)
    n_zones = len(test_combination)


    zone_combinations = list(kombi_results.keys())

    possibilities = []
    for combo in zone_combinations:
        if determine_takst_region(combo) == takst:
            if n_zones - 3 < len(combo) < n_zones + 3:
                symdiff = set(combo).symmetric_difference(test_combination)
                if len(symdiff) <= 2:
                    possibilities.append(combo)
    return sorted(possibilities, key=lambda x: len(x), reverse=True)


def _join_note_p(notelist: List[str]) -> str:

    return ''.join(
        (''.join(j) + r'->' if i!=len(notelist)-1 else ''.join(j)
         for i, j in enumerate(notelist)
         )
        )

def _load_kombi_shares(year: int, model: int) -> dict:

    fp = os.path.join(
        THIS_DIR,
        '__result_cache__', f'{year}',
        'pendler', f'pendlerchosenzones{year}_model_{model}.csv'
        )

    df = pd.read_csv(fp, index_col=0)
    df.rename(columns={'S-tog': 'stog'}, inplace=True)

    d = df.to_dict(orient='index')

    return {ast.literal_eval(k): v for k, v in d.items()}

def _load_nzone_shares(year: int, model: int):


    takst_map = {
        'dsb': 'dsb',
        'th': 'hovedstad',
        'ts': 'sydsjælland',
        'tv': 'vestsjælland'
        }

    filedir = os.path.join(
        THIS_DIR,
        '__result_cache__',
        f'{year}', 'pendler'
        )
    files = glob.glob(os.path.join(filedir, '*.csv'))
    kombi = [x for x in files if 'kombi_paid_zones' in
             x and f'model_{model}' in x]

    out = {}
    for file in kombi:
        df = pd.read_csv(file, index_col=0)
        d = df.to_dict(orient='index')
        for k, v in takst_map.items():
            if v in file:
                out[k] = d
                break

    return out


def _load_zone_relation_results(kombi_results):
    """load the relations and results for the zone relations and merge them

    :param kombi_results: [description]
    :type kombi_results: [type]
    :return: [description]
    :rtype: [type]
    """

    zone_relations = _load_zone_relations()

    test = {}
    for x in zone_relations:
        try:
            rel = (x['StartZone'], x['DestinationZone'], x['PaidZones'])
            validzones = x['ValidZones']
        except Exception as e:
            continue
        test[rel] = validzones

    zone_relation_results = {}
    for k, v in test.items():
        try:
            zone_relation_results[k] = kombi_results[v]
        except KeyError:
            pass
    return zone_relation_results

def _valid_zones_to_paid():
    zone_relations = _load_zone_relations()
    d = {}
    for x in zone_relations:
        try:
            d[x['ValidZones']] = x['PaidZones']
        except KeyError:
            pass
    return d

def _user_share_dict_df(usershares: USER_SHARES):
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

    return ushares


def _match_user_specific_results(
        kombi_products: pd.core.frame.DataFrame,
        usershares: USER_SHARES,
        year: int,
        model: int
        ):

    usershares_df = _user_share_dict_df(usershares)

    merged = pd.merge(
        kombi_products, usershares_df,
        on=['EncryptedCardEngravedID', 'SeasonPassID'],
        how='left'
        )
    merged = merged.fillna(0)
    missed = merged.query("n_trips == 0").copy()

    kombi_match = merged.query("n_trips != 0").copy()
    kombi_match['note'] = 'user_period_shares'

    missed = missed[kombi_products.columns]

    missed = _match_pendler(missed, year, model)
    return kombi_match, missed


def _match_pendler_record(
        record: Tuple[Any, ...],
        kombi_results,
        zone_relation_results,
        paid_zones_results,
        min_trips: int
        ):
    """match a sales record to a result

    :param record: [description]
    :type record: Tuple[Any, ...]
    :param kombi_results: [description]
    :type kombi_results: [type]
    :param zone_relation_results: [description]
    :type zone_relation_results: [type]
    :param paid_zones_results: [description]
    :type paid_zones_results: [type]
    :param min_trips: the minimum number of samples allowed
    :type min_trips: int
    :raises ValueError: if the startzone can't be determined
    :return: [description]
    :rtype: Dict
    """

    chosen_zones = ast.literal_eval(record.valgtezoner)
    try:
        takst = record.takstsæt.lower()
    except AttributeError:
        takst = determine_takst_region(chosen_zones)
    try:
        start = int(record.startzone)
    except ValueError:
        if '/' in record.startzone:
            start = int(record.startzone.split('/')[1])
        else:
            raise ValueError("Can't determine startzone")

    end = int(record.slutzone)
    paid = int(record.betaltezoner)

    note = []
    flag = False
    if chosen_zones:

        mro = ['kombimatch', f'kombi_paid_zones_{takst}']
        if 1002 in chosen_zones and 1001 not in chosen_zones:
            note.append('INVALID_KOMBI')
            flag = True

        if flag:
            mro = ['kombimatch', 'closekombi', f'kombi_paid_zones_{takst}']

        for method in mro:
            note.append(method)
            if method == 'closekombi':
                chosen_zones = _get_closest_kombi(chosen_zones)
            r = kombi_results.get(chosen_zones, {})

            if r and r['n_trips'] >= min_trips:
                break
            else:
                r = paid_zones_results[takst].get(paid, {})

        if not r:
            note.append('from_to_zones')
            from_to_results = {
                k[:2]:v for k, v in zone_relation_results.items() if
                k[0]==start and k[1]==end
                }
            r = from_to_results.get((start, end), {})

        if not r:
            note.append('closest_chosenzones_match')
            possibilities = _get_closest_chosen_zones(kombi_results, chosen_zones)
            for pos in possibilities:
                r = kombi_results.get(pos, {})
                if r:
                    note.append(f'_{pos}')
                    break

    else:
        mro = ['zonerelation_match', f'kombi_paid_zones_{takst}']
        for method in mro:
            note.append(method)
            r = zone_relation_results.get((start, end, paid), {})

            if r and r['n_trips'] >= min_trips:
                break
            else:
                r = paid_zones_results[takst].get(paid, {})

    if not r or r['n_trips'] < min_trips:
        note.append(f'kombi_paid_all_zones_{takst}')
        r = paid_zones_results[takst].get(99, {})

    out = r.copy()
    out['note'] = _join_note_p(note)
    return out

def _match_pendler(pendler_df, year, model):

    pendler_df.loc[:, 'NR'] = list(range(len(pendler_df)))

    sub_tuples = list(pendler_df.itertuples(index=False, name='Sale'))

    kombi_results = _load_kombi_shares(year, model)
    zone_relation_results = _load_zone_relation_results(kombi_results)
    paid_zones_results = _load_nzone_shares(year, model)

    bad = set()
    out = {}
    for record in sub_tuples:
        try:
            result =  _match_pendler_record(
                record,
                kombi_results,
                zone_relation_results,
                paid_zones_results,
                1
                )
            out[record.NR] = result
        except:
            bad.add(record.NR)
    # merge into _result_frame func
    out_frame = pd.DataFrame.from_dict(out).T
    out_frame.index.name = 'NR'
    out_frame = out_frame.reset_index()

    output = pd.merge(pendler_df, out_frame, on='NR', how='left')

    output.note = output.note.fillna('no_result')
    output = output.fillna(0)

    output = output.drop(['NR','key'], axis=1)

    return output


def _process_pendler_df(period_products, zone_path):
    try:
        period_products.Price = \
        period_products.Price.str.replace(',','', regex=False
        ).str.replace('.','', regex=False).astype(float) / 100
    except AttributeError:
        pass

    pendler_product_zones = _load_process_zones(zone_path)

    keys = dict(zip(zip(pendler_product_zones.EncryptedCardEngravedID,
               pendler_product_zones.SeasonPassID), pendler_product_zones.valgtezoner))

    period_products['key'] = period_products.loc[
        :, ('EncryptedCardEngravedID', 'SeasonPassID')
        ].apply(tuple, axis=1)

    paid_map = _valid_zones_to_paid()
    paid_map = {str(k): v for k, v in paid_map.items()}
    period_products.loc[:, 'valgtezoner'] = period_products.loc[:, 'key'].map(keys)

    period_products.loc[:, 'valgtezoner'] = period_products.loc[:, 'valgtezoner'].astype(str)
    period_products.loc[:, 'betaltezoner'] = period_products.loc[:, 'valgtezoner'].map(paid_map)
    period_products.loc[:, 'betaltezoner'] = period_products.loc[:, 'betaltezoner'].fillna(0)
    period_products.rename(
        columns={
            'ZoneNrLow': 'startzone',
            'ZoneNrHigh': 'slutzone',
            'PsedoFareset': 'takstsæt'
            }, inplace=True)

    period_products.takstsæt = period_products.takstsæt.str.lower()
    takst_map = {'hovedstaden': 'th', 'sjælland': 'dsb', 'sydsjælland': 'ts', 'vestsjælland': 'tv'}

    period_products.takstsæt = period_products.takstsæt.map(takst_map)

    return period_products

def _determine_city_note(chosen_zones):

    city_zones = (1001, 1002, 1003, 1004)

    try:
        chosen_zones = ast.literal_eval(chosen_zones)
    except ValueError:
        pass

    if not chosen_zones:
        return ''

    if all(x in city_zones for x in chosen_zones):
        return 'all_city'
    if any(x in city_zones for x in chosen_zones):
        return 'with_city'

    return 'no_city'

def add_city_note(df):
    df['city_note'] = df.loc[:, 'valgtezoner'].apply(lambda x: _determine_city_note(x))
    return df

def _set_takst(valgtezoner):

    chosen = ast.literal_eval(valgtezoner)

    if all(x < 1100 for x in chosen):
        return 'TH'
    if all(1100 < x < 1200 for x in chosen):
        return 'TV'
    if all(1200 < x < 1300 for x in chosen):
        return 'TS'
    return 'DSB'

def make_output(usershares, product_path, zone_path, model, year):
    """

    :param usershares: DESCRIPTION
    :type usershares: TYPE
    :param product_path: DESCRIPTION
    :type product_path: TYPE
    :param zone_path: DESCRIPTION
    :type zone_path: TYPE
    :param model: DESCRIPTION
    :type model: TYPE
    :param year: DESCRIPTION
    :type year: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    # this might need to change based on input data structure
    period_products = pd.read_csv(
        product_path, sep=';', encoding='utf8'
        )

    period_products = _process_pendler_df(period_products, zone_path)

    kombi_products = period_products.loc[
        period_products.loc[:, 'ProductName'].str.lower().str.contains('kombi')
        ]

    kombi_match, missed = _match_user_specific_results(
        kombi_products, usershares, year, model
        )

    # =============================================================================
    # end of direct card match
    # =============================================================================
    pendler = period_products.loc[~
        period_products.loc[:, 'ProductName'].str.lower().str.contains('kombi')
        ].copy()


    pendler_results = _match_pendler(pendler, year, model)

    pendler = pendler.drop('key', axis=1)
    initial_columns = list(pendler.columns)
    kombi_match.columns = kombi_match.columns.astype(str)
    missed.columns = missed.columns.astype(str)
    pendler_results.columns = pendler_results.columns.astype(str)


    final = pd.concat([kombi_match, missed, pendler_results], axis=0)

    stats_columns = ['n_trips', 'n_users', 'n_period_cards', 'note']
    operator_columns = [
        x for x in final.columns if x not in initial_columns
        and x not in stats_columns
        ]

    andel_columns = []
    for col in operator_columns:
        new_col = f'{col}_andel'
        andel_columns.append(new_col)
        final.loc[:, f'{col}_andel'] = \
            final.loc[:, 'Price'] * final.loc[:, col]

    col_order = initial_columns + operator_columns + andel_columns + stats_columns
    final = final[col_order]

    final = add_city_note(final)
    try:
        final = final.drop('key', axis=1)
    except KeyError:
        pass

    final['PsedoFareset'] = final.valgtezoner.apply(_set_takst)

    return final

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


def fetch_trip_results(db_path, tofetch):

    with lmdb.open(db_path) as env:
        final = {}
        with env.begin() as txn:
            for card_season, trips in tqdm(tofetch.items()):
                all_trips = []
                for trip in trips:
                    t = txn.get(bytes(str(trip), 'utf-8'))
                    if t is not None:
                        all_trips.append(t.decode('utf-8'))
                all_trips = tuple(
                    ast.literal_eval(x) for x in all_trips
                    if x not in TRIP_ERRORS
                    )
                card_season = ast.literal_eval(card_season)
                final[card_season] = get_user_shares(all_trips)

    return final

def main():

    # Use zero price
    parser = TableArgParser(
        'year', 'zones', 'products'
        )
    args = parser.parse()

    year = args['year']
    paths = db_paths(find_datastores(), year)
    db_path = paths['calculated_stores']

    products = args['products']
    zones = args['zones']

    stores = paths['store_paths']
    valid_kombi_store = paths['kombi_valid_trips']

    userdata = PendlerKombiUsers(
        year,
        products_path=products,
        product_zones_path=zones,
        min_valid_days=0
        ).get_data()

    userdata = process_user_data(userdata)
    kombi_trips = load_valid(valid_kombi_store)
    zero_travel_price = find_no_pay(stores, year, 4)

    valid = {
        k: set(v).intersection(zero_travel_price) for
        k, v in kombi_trips.items()
        }
    for model in [1, 2, 3, 4, 5, 6]:
        path = db_path + f'_model_{model}'
        results = fetch_trip_results(path, valid)

        final = make_output(results,
                    products,
                    zones,
                    model,
                    year)
        fp = THIS_DIR / '__result_cache__'/ f'{year}' /f'rejsekort_shares_model{model}.csv'

        final.to_csv(fp, index=False, encoding='utf8')


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
