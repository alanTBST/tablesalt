# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:18:51 2020

@author: alkj
"""
import ast
import glob
import os
import pickle
from itertools import chain
from operator import itemgetter
from typing import AnyStr, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from tablesalt.preprocessing.parsing import TableArgParser

THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent


# TODO maybe put this in a config file
LOCATIONS = {
    'automat': 'metro',
    'nautila': 'dsb',
    'lokaltog-automater i 25 nordsjællandske togsæt': 'movia',
    'enkeltbilletter bus': 'movia'
    }

RESULT_MAP = {'dsb': ('D**', 'D*', 'D'),
              'metro': ('Metro',),  
              'movia': ('Movia_H', 'Movia_S', 'Movia_V')}

def _proc_sales(frame):

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].apply(
        lambda x: tuple(sorted(ast.literal_eval(x)))
        )

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].astype(str)
    return frame


def _load_sales_data(year: int) -> pd.core.frame.DataFrame:

    fp = os.path.join(
        THIS_DIR, '__result_cache__', 
        f'{year}', 'preprocessed', 
        'mergedsales.csv'
        )

    df = pd.read_csv(fp, low_memory=False)

    return _proc_sales(df)

def _get_single_results(year):
    
    fp = os.path.join(
        THIS_DIR, '__result_cache__', 
        f'{year}', 'preprocessed'
        )
    singles = glob.glob(os.path.join(fp, 'single*'))
    
    out = {}
    for i in range(3):
        fmatch = [x for x in singles if f'r{i}' in x]
        times = [os.path.getmtime(x) for x in fmatch]
        min_id = np.argmin(times) # find latest entry
        fp = fmatch[min_id]
        with open(fp, 'rb') as f:
            res = pickle.load(f)
        out[f'rabat{i}'] = res
            
    return out

def _sales_ref(frame: pd.core.frame.DataFrame) -> Dict[str, Tuple[int, ...]]:
    products = set(frame.loc[:, 'produktnavn'])

    sales_idxs = {}

    for prod in products:
        sales_idxs[prod] = tuple(
            frame.query("produktnavn == @prod").loc[:, 'NR']
            )

    return sales_idxs

def _location_ref(
        frame: pd.core.frame.DataFrame
        ) -> Dict[str, Tuple[int, ...]]:

    sources = set(frame.loc[:, 'salgsmedie'])

    location_idxs = {}

    for src in sources:
        location_idxs[src] = tuple(
            frame.query("salgsmedie == @src").loc[:, 'NR']
            )

    return location_idxs


def _get_location_sales(location_idxs, sales_idxs):
    
    wanted_start_locations = {
        k: v for k, v in location_idxs.items() if k in LOCATIONS
        }
    
    single = set(sales_idxs['enkeltbillet']) | \
        set(sales_idxs['lang enkeltbillet'])
    location_sales = {}
    for k, v in wanted_start_locations.items():
        location_sales[k] = set(v).intersection(single)
          
    return location_sales

def _get_location_results(location: str, results):
    
    operator = LOCATIONS[location]    
    result_keys = set(RESULT_MAP[operator])
    result_keys.add('all')
      
    res_dict = {k: {k1: v1 for k1, v1 in v.items() if k1 in result_keys} 
                for k, v in results.items()}
    
    return res_dict

def _method_resolution_operator(res, length):
    
    mro = []
    any_start = []
    paid = []
    paid_any = []
    
    for k, v in res.items():
        if length == 'short':
            start_op = [(k, x, 'short_ring') for x in v if x != 'all']  
            start_any = [(k, x, 'short_ring')for x in v if x == 'all'] 
        
        if length == 'long':
            start_op = [(k, x, 'long') for x in v if x != 'all'] + \
                [(k, x, 'long')for x in v if x == 'all']                
            
            start_any = [(k, x, 'long_ring') for x in v if x != 'all'] + \
                [(k, x, 'long_ring')for x in v if x == 'all'] 
                    
        start_op_paid = [(k, x, 'paid_zones') for x in v if x != 'all']        
        start_any_paid = [(k, x, 'paid_zones') for x in v if x == 'all']
        
        mro.extend(start_op)   
        any_start.extend(start_any)
        paid.extend(start_op_paid)
        paid_any.extend(start_any_paid)
    
    if length == 'long':
        mro = sorted(mro, key=itemgetter(1, 0))
    mro = mro + any_start + paid + paid_any
                
    return mro

def _method_resolution_any(res, length):
    
    
    return 


def _match_single_any():
    
    
    
    return 


def _filter_mro(record, mro):
    
    if record.takstsæt == 'th':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_V' not in x]
    elif record.takstsæt == 'tv':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_H' not in x]
    elif record.takstsæt == 'ts':
        mro = [x for x in mro if 'Movia_H' not in x and 'Movia_V' not in x]
    
    return mro

def _match_single_operator(record, res, mro, trip_threshhold=0):
    
    
    mro = _filter_mro(record, mro)
    
    try:
        ring_ticket = int(record.startzone), int(record.betaltezoner)
        ft_ticket = int(record.startzone), int(record.slutzone)
    except ValueError:
        return {}
    
    note = []
    for method in mro:
        note.append(method)
        try:
            d = res[method[0]][method[1]][method[2]]
        except KeyError:
            continue
        
        if 'short_ring' in method or 'long_ring' in method:
            r = d.get(ring_ticket, {})
        elif 'long' in method:
            r = d.get(ft_ticket, {})
        else:
            r = d.get(ring_ticket[1], {})            
                
        if not r:
            continue
        if r and r['n_trips'] == trip_threshhold:
            continue
        if r is not None:
            break

    else:        
        r = {}   
    r['note'] = ''.join(
        ('_'.join(j) + r'->' if i!=len(note)-1 else '_'.join(j) 
         for i, j in enumerate(note)
         )
        )
    return r

def _location_merge(location_sales, data, single_results):
    
    final = {}
    for loc in LOCATIONS:
        location_nr = location_sales[loc]
        sub_data = data.query("NR in @location_nr")
        sub_tuples = list(sub_data.itertuples(index=False, name='Sale'))
        
        res = _get_location_results(loc, single_results)   
        mro_short = _method_resolution_operator(res, 'short')
        mro_long = _method_resolution_operator(res, 'long')
         
        out = {}
        for record in sub_tuples:
            if record.betaltezoner <= 8:
                out[record.NR] = _match_single_operator(record, res, mro_short)
            else:
                out[record.NR] = _match_single_operator(record, res, mro_long)
                
        out_frame = pd.DataFrame.from_dict(out).T
        out_frame.index.name = 'NR'
        out_frame = out_frame.reset_index()
        
        test_out = pd.merge(sub_data, out_frame, on='NR', how='left')
        test_out.note = test_out.note.fillna('')
        test_out = test_out.fillna(0)
        
        main_cols = list(sub_data.columns)
        new = [x for x in test_out if x not in main_cols]
        ops = [x for x in new if x not in ('note', 'n_trips')]
        
        out_cols = main_cols + sorted(ops) + ['n_trips', 'note']
        test_out = test_out[out_cols]
        final[loc] = test_out
        # test_out.to_csv(f'{loc}_single_new.csv', index=False)
               
    return final


def main():
    
    parser = TableArgParser('year')
    args = parser.parse()
    year = args['year']
    
    year = 2019
       
    sales_data = _load_sales_data(year)
    sales_idxs = _sales_ref(sales_data)
    location_idxs = _location_ref(sales_data)
    location_sales = _get_location_sales(location_idxs, sales_idxs)
    
    return 
# =============================================================================
# Loading functions
# =============================================================================
# def _check_names(frame):

#     frame.rename(columns={
#         'start_zone': 'startzone',
#         'StartZone': 'startzone',
#         'DestinationZone': 'slutzone',
#         'n_zones': 'betaltezoner',
#         'PaidZones': 'betaltezoner',
#         'end_zone': 'slutzone',
#         'S-tog': 'stog',
#         'DSB': 'dsb',
#         'Movia_H': 'movia',
#         'Metro': 'metro',
#         'First': 'first'
#         }, inplace=True)
#     return frame

# def _stringify_merge_cols(frame):

#     string_cols = ['betaltezoner', 'slutzone', 'startzone']
#     for col in string_cols:
#         try:
#             frame.loc[:, col] = \
#                 frame.loc[:, col].astype(str)
#         except KeyError:
#             pass

#     return frame

# def _load_ringzone_shares() -> pd.core.frame.DataFrame:

#     filename = r'__result_cache__/2019/single/start_all_short_ring_2019_r0.csv'

#     df = pd.read_csv(filename)
#     df = _check_names(df)
#     df = _stringify_merge_cols(df)

#     return df

# def _load_long_shares(ring: Optional[bool] = False):

#     if not ring:
#         filename = r'__result_cache__/2019/single/start_all_long_2019_r0.csv'
#     else:
#         filename = r'__result_cache__/2019/single/start_all_long_ring_2019_r0.csv'
#     df = pd.read_csv(filename)
#     df = _check_names(df)
#     df = _stringify_merge_cols(df)

#     return df


# def _load_operator_shares(
#         operator: str,
#         length: str,
#         ring: Optional[bool] = False
#         ) -> pd.core.frame.DataFrame:

#     operator = operator.lower()
#     length = length.lower()

#     filedir = os.path.join('__result_cache__', '2019', 'single')

#     files = glob.glob(os.path.join(filedir, '*.csv'))
#     files = [x for x in files if 'start_' in x and 'all' not in x]
#     wanted = [x for x in files if operator in x.lower() and length in x.lower()]
#     wanted = [x for x in wanted if '_r0' in x]

#     if not ring:
#         wanted = [x for x in wanted if 'ring' not in x.lower()]
#     else:
#         wanted = [x for x in wanted if 'ring' in x.lower()]

#     if not wanted:
#         raise ValueError(
#             f"no files match {operator} and {length} and ring={ring}"
#             )

#     frames = []
#     for f in wanted:
#         df = pd.read_csv(f)
#         df = _check_names(df)
#         if 'Movia_H' in f:
#             df = df.query("startzone < 1100")
#         elif 'Movia_S' in f:
#             df = df.query("startzone > 1200")
#         else:
#             df = df.query("1100 < startzone < 1200")

#         df = _stringify_merge_cols(df)
#         frames.append(df)

#     return pd.concat(frames)



# =============================================================================
# Pendler tickets
# =============================================================================

def _load_kombi_shares() -> pd.core.frame.DataFrame:

    filename = r'__result_cache__/2019/pendler/pendlerkeys2019.csv'

    df = pd.read_csv(filename, index_col=0)
    df.index.name = 'valgtezoner'
    df = df.reset_index()
    df.rename(columns={'S-tog': 'stog'}, inplace=True)

    return df

def _load_kombi_map_shares() -> pd.core.frame.DataFrame:

    filename = r'__result_cache__/2019/pendler/zone_relation_keys2019.csv'
    df = pd.read_csv(filename)
    df = _check_names(df)
    df.loc[:, 'betaltezoner'] = df.loc[:, 'betaltezoner'].fillna(0)
    df.loc[:, 'betaltezoner'] = df.loc[:, 'betaltezoner'].astype(int)

    df = df.drop(['ValidityZones', 'ValidZones'], axis=1)

    df = _stringify_merge_cols(df)

    return df


def _load_nzone_shares(takst: str, year: int):
    
    takst_map = {'dsb': 'dsb', 'th': 'hovedstad', 'ts': 'sydsjælland', 'tv': 'vestsjælland'}
    takst = takst_map[takst]
    
    filedir = os.path.join(THIS_DIR, '__result_cache__', f'{year}', 'pendler')
    files = glob.glob(os.path.join(filedir, '*.csv'))
    kombi = [x for x in files if 'kombi_paid' in x]
    region = [x for x in kombi if takst in x][0]

    df = pd.read_csv(region, index_col=0)
    df.index.name = 'betaltezoner'
    df = df.reset_index()
    df = _check_names(df)
    df = _stringify_merge_cols(df)
    return df

def _chosen_fallback(nullframe, takst, year):

    nzones = _load_nzone_shares(takst, year)

    nullframe = nullframe.drop([
        'movia', 'stog', 'metro',
        'dsb', 'first', 'n_users',
        'n_period_cards', 'n_trips'], axis=1
        )

    nzone_output = pd.merge(
        nullframe, nzones,
        on=['betaltezoner'],
        how='left'
        )
    nzone_output.loc[nzone_output.loc[:, 'valgtezoner'].apply(
        lambda x: '1002' in x and '1001' not in x), 'Note'] = \
        nzone_output.loc[nzone_output.loc[:, 'valgtezoner'].apply(
            lambda x: '1002' in x and '1001' not in x), 'Note'].apply(
                lambda x: x + r'/kombi_not_allowed')

    nzone_output = _extend_note(
        nzone_output, 'kombi_nzones',
        'kombi_nzones/no_trips')

    return nzone_output


def _mappable_fallback():

    return

def _sub_kombi(sub_frame, takst, year):

    cols = ['movia', 'stog', 'metro',
            'dsb', 'first', 'n_users',
            'n_period_cards', 'n_trips']

    if any(x in sub_frame.columns for x in cols):
        sub_frame = sub_frame.drop(cols, axis=1)

    chosen_frame = sub_frame.loc[sub_frame.loc[:, 'valgtezoner'] != '()']
    kombi_results = _load_kombi_shares()
    chosen_merge = pd.merge(
        chosen_frame, kombi_results,
        on='valgtezoner',
        how='left'
        )

    chosen_merge = _extend_note(chosen_merge, 'kombi_match', 'no_kombi_match')
    missed = chosen_merge[chosen_merge.n_trips.isnull() |
                          (chosen_merge.n_trips == 0)]
    if not missed.empty:
        chosenfall =  _chosen_fallback(missed, takst, year)
        chosen_merge = pd.concat(
            [chosen_merge[chosen_merge.n_trips.notnull()], chosenfall]
            )

    return chosen_merge

def _kombi_mappable_pendler(sale_id, frame, takst, year):

    sub_frame = frame.query("NR in @sale_id")
    sub_frame = sub_frame.query("takstsæt == @takst")
    # sub_frame = sub_frame.query("betaltezoner < 90") # ensure not all zones
    sub_frame = _stringify_merge_cols(sub_frame)

    chosen_frame = sub_frame.loc[sub_frame.loc[:, 'valgtezoner'] != '()']
    kombi_results = _load_kombi_shares()
    chosen_merge = pd.merge(
        chosen_frame, kombi_results,
        on='valgtezoner',
        how='left'
        )

    chosen_merge = _add_note(chosen_merge, 'kombi_match', 'no_kombi_match')
    missed = chosen_merge[chosen_merge.n_trips.isnull() |
                          (chosen_merge.n_trips == 0)]

    try:
        if not missed.empty:
            chosenfall =  _chosen_fallback(missed, takst, year)
            chosen_merge = pd.concat(
                [chosen_merge[chosen_merge.n_trips.notnull()], chosenfall]
                )
    except FileNotFoundError:
        pass

    mappable_frame = sub_frame.loc[sub_frame.loc[:, 'valgtezoner'] == '()']
    mappable_frame = mappable_frame.loc[
        mappable_frame.loc[:, 'startzone'].apply(lambda x: str(x)[1:]) != '000'
        ]

    mappable_frame = _stringify_merge_cols(mappable_frame)

    zone_rel = _load_kombi_map_shares()
    mappable_merge = pd.merge(
        mappable_frame, zone_rel,
        on=['startzone', 'slutzone', 'betaltezoner'],
        how='left'
        )

    mappable_merge = _add_note(
        mappable_merge,
        'zone_relation_match',
        'no_zone_relation_match'
        )
    missed = mappable_merge[(mappable_merge.n_trips.isnull()) |
                            (mappable_merge.n_trips == 0)]

    try:
        if not missed.empty:
            missed.loc[:, 'valgtezoner'] = missed.loc[
                :, ('startzone', 'slutzone')
                ].astype(int).apply(tuple, axis=1)
            missed.loc[:, 'valgtezoner'] = missed.loc[:, 'valgtezoner'].apply(
                lambda x: str(tuple(sorted(x)))
                )
            recursive_fallback = _sub_kombi(missed, takst, year)
            mappable_merge = pd.concat(
                [mappable_merge[
                    mappable_merge.n_trips.notnull()], recursive_fallback
                    ]
                )
    except FileNotFoundError:
        pass

    out = pd.concat([chosen_merge, mappable_merge])
    out = out.sort_values(['NR', 'n_trips'])
    out = out.drop_duplicates('NR', keep='last')
    return out



def _nzone_pendler(
        sale_id: Dict,
        frame: pd.core.frame.DataFrame,
        takst: str,
        year: int
        ) -> pd.core.frame.DataFrame:

    sub_frame = frame.query("NR in @sale_id")
    sub_frame = sub_frame.query("takstsæt == @takst")
    sub_frame = sub_frame.loc[
        (sub_frame.loc[:, 'startzone'].apply(lambda x: str(x)[1:]) == '000') |
        (sub_frame.loc[:, 'slutzone'].apply(lambda x: str(x)[1:]) == '000')
        ]
    sub_frame = _stringify_merge_cols(sub_frame)

    nzone_shares = _load_nzone_shares(takst, year)

    nzone_output = pd.merge(
        sub_frame, nzone_shares,
        on=['betaltezoner'],
        how='left'
        )

    return nzone_output

def _process_pendler(
        sales_idxs: Dict,
        frame: pd.core.frame.DataFrame,
        takst: str,
        year:int) -> pd.core.frame.DataFrame:

    # TODO make these a user input option

    kombi_pendler_ids = \
        sales_idxs.get('pendlerkort', ())  + \
            sales_idxs.get('pensionistkort', ())  + \
                sales_idxs.get('ungdomskort vu', ())  + \
                    sales_idxs.get('ungdomskort uu', ()) + \
                        sales_idxs.get('flexcard', ())
    kombi_pendler = _kombi_mappable_pendler(
        kombi_pendler_ids, frame, takst, year
        )
    kombi_ids = set(kombi_pendler.loc[:, 'NR'])

    # TODO make these a user input option


    # default_options = {
    # 'pendlerkort', 'erhvervskort', 'virksomhedskort', 'pensionistkort',
    # 'ungdomskort xu','flexcard 7 dage', 'ungdomskort vu',
    # 'flexcard', 'ungdomskort uu',}

    general_pendler_ids = \
        sales_idxs.get('pendlerkort', ()) + \
            sales_idxs.get('erhvervskort', ())  + \
                sales_idxs.get('virksomhedskort', ())  + \
                    sales_idxs.get('pensionistkort', ())  + \
                        sales_idxs.get('ungdomskort xu', ())  + \
                            sales_idxs.get('flexcard 7 dage', ()) + \
                                sales_idxs.get('flexcard', ()) + \
                                    sales_idxs.get('ungdomskort vu', ())  + \
                                    sales_idxs.get('ungdomskort uu', ())


    general_pendler_ids = tuple(
        x for x in general_pendler_ids if x not in kombi_ids
        )

    try:
        gen_pendler = _nzone_pendler(general_pendler_ids, frame, takst, year)
        gen_pendler = _add_note(
            gen_pendler,
            'kombi_any_nzone',
            'kombi_any_nzone/no_match'
            )
        return pd.concat([kombi_pendler, gen_pendler])
    except FileNotFoundError:
        return kombi_pendler

    # fallback all kombi TH

def _load_rabatzero_shares():

    filename = r'__result_cache__/2019/other/citypass_shares.csv'
    df = pd.read_csv(filename, index_col=0)
    df.index.name = 'betaltezoner'
    df = df.reset_index()

    # TODO change this output format
    df.loc[:, 'betaltezoner'] = df.loc[:, 'betaltezoner'].replace(
        {'citypass_large': '99',
         'citypass_small': '4'}
        )
    df = _check_names(df)
    df = _stringify_merge_cols(df)

    return df

def _inner_city(sales_idxs: Dict,
                frame: pd.core.frame.DataFrame,
                takst: str):

    # default values
    th = {'city pass small',
          'citypass small 24 timer',
          'citypass small 120 timer',
          'citypass small 48 timer',
          'citypass small 72 timer',
          'citypass small 96 timer',
          'mobilklippekort'}

    th_ids = set().union(*[set(sales_idxs.get(x, set())) for x in th])
    sub_frame = frame.query("NR in @th_ids")
    sub_frame = sub_frame.query("takstsæt == @takst")
    sub_frame = sub_frame.loc[
        ~((sub_frame.loc[:, 'produktnavn'] == 'mobilklippekort') &
         (sub_frame.loc[:, 'betaltezoner'] != 4)) ]

    th_shares = _load_rabatzero_shares()
    th_shares = th_shares[th_shares.loc[:, 'betaltezoner'] != '99']

    length = len(sub_frame)
    merge_frame = pd.concat([th_shares]*length, ignore_index=True)
    merge_frame = merge_frame.drop('betaltezoner', axis=1)
    merge_frame['NR'] = list(sub_frame['NR'])
    out = pd.merge(sub_frame, merge_frame, on='NR')
    out = _add_note(out, 'all trips 01-04', 'all trips 01-04/no_trips')

    return out


def _all_sub_takst(
        sales_idxs: Dict,
        frame: pd.core.frame.DataFrame,
        takst: str):

    # default values
    th = {'institutionskort, 20 børn', 'travel pass',
          'kulturnatten', 'citypass large 120 timer',
          'citypass large 24 timer',
          'børnealderskompensation', 'city pass large',
          'citypass large 96 timer',
          'citypass large 72 timer',
          'copenhagen card',
          'citypass large 48 timer', 'skoleklassekort',
          'institutionskort, 15 børn',
          'blindekort',
          'off peak-kompensation',
          'mobilklippekort',
          'turistbillet',
          'print-selv-billet',
          'enkeltbillet refusion'}

    th_ids = set().union(*[set(sales_idxs.get(x, set())) for x in th])

    sub_frame = frame.query("NR in @th_ids")
    sub_frame = sub_frame.loc[
        (sub_frame.loc[:, 'takstsæt'] == takst) |
        (sub_frame.loc[:, 'produktnavn'].str.contains('city'))
        ]
    sub_frame = sub_frame.loc[
        ~((sub_frame.loc[:, 'produktnavn'] == 'mobilklippekort') &
         (sub_frame.loc[:, 'betaltezoner'] == 4))
        ]
    sub_frame = sub_frame.loc[
        ~((sub_frame.loc[:, 'produktnavn'] == 'turistbillet') &
         (sub_frame.loc[:, 'betaltezoner'] != 99))
        ]
    sub_frame = sub_frame.loc[
        ~((sub_frame.loc[:, 'produktnavn'] == 'print-selv-billet') &
         (sub_frame.loc[:, 'betaltezoner'] != 99))
        ]

    length = len(sub_frame)

    th_shares = _load_rabatzero_shares()
    th_shares = th_shares[th_shares.loc[:, 'betaltezoner'] == '99']
    merge_frame = pd.concat([th_shares]*length, ignore_index=True)
    merge_frame = merge_frame.drop('betaltezoner', axis=1)
    merge_frame['NR'] = list(sub_frame['NR'])
    out = pd.merge(sub_frame, merge_frame, on='NR')
    out = _add_note(
        out, 'all rabat0trips TH',
        'all  rabat0trips TH/no_trips'
        )

    return out


def _process_other(sales_idxs, frame, takst):

    # all_zones = _all_sub_takst(sales_idxs, frame, takst)
    cph = _inner_city(sales_idxs, frame, takst)
    return cph
    # return pd.concat([all_zones, cph])

def main():
    """main function"""
    parser = TableArgParser('year')
    args = parser.parse()
    year = args['year']
    
    year = 2019
       
  
    sales_data = _load_sales_data(year)
    sales_idxs = _sales_ref(sales_data)

# =============================================================================
    single_output = []
    for takst in ['th', 'ts', 'dsb', 'tv']:
        single = _process_single(sales_idxs, sales_data, takst)
        single_output.append(single)
    single_output = pd.concat(single_output)
# =============================================================================

    pendler_output = []
    for takst in ['th', 'ts', 'dsb', 'tv']:
        pendler = _process_pendler(sales_idxs, sales_data, takst, year)
        pendler_output.append(pendler)
    pendler_output = pd.concat(pendler_output)
# =============================================================================
    other_output_h = _process_other(sales_idxs, sales_data, 'th')
# =============================================================================


    found = set(single_output.NR).union(set(pendler_output.NR)).union(
        set(other_output_h.NR))
    missing = sales_data.query("NR not in @found")

    final = pd.concat([single_output, pendler_output, other_output_h, missing])
    final = final.sort_values(['NR'])

    cols = [
        x for x in final.columns if
        'Unnamed' not in x and x not in ('tup', 'ringdist')
        ]
    final = final[cols]
    final = final.drop_duplicates()
# =============================================================================

    return final


# if __name__ == "__main__":
#       out = main()
