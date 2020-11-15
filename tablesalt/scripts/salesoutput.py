# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:18:51 2020

@author: alkj
"""
import ast
import glob
import os
import pickle
import pkg_resources
from itertools import chain
from operator import itemgetter
from typing import AnyStr, Dict, Tuple, Optional, List
from pathlib import Path

import msgpack
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

USE_LOCATIONS = False

RESULT_MAP = {'dsb': ('D**', 'D*', 'D'),
              'metro': ('Metro',),  
              'movia': ('Movia_H', 'Movia_S', 'Movia_V')}

def _proc_sales(frame):

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].apply(
        lambda x: tuple(sorted(ast.literal_eval(x)))
        )

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].astype(str)
    
    frame.loc[:, 'slutzone'] = frame.loc[:, 'slutzone'].fillna('1000')
    frame.loc[:, 'startzone'] = frame.loc[:, 'startzone'].fillna('1000')

    return frame


def _load_sales_data(year: int) -> pd.core.frame.DataFrame:

    fp = os.path.join(
        THIS_DIR, '__result_cache__', 
        f'{year}', 'preprocessed', 
        'mergedsales.csv'
        )

    df = pd.read_csv(fp, low_memory=False)
    
    return _proc_sales(df)

def _get_single_results(year: int,  model: int):

    
    fp = os.path.join(
        THIS_DIR, 
        '__result_cache__', 
        f'{year}', 
        'preprocessed'
        )
    singles = glob.glob(os.path.join(fp, 'single*'))

    singles = [x for x in singles if f'model_{model}' in x]
    
    out = {}
    for i in range(3): # using r0, r1, r2

        fmatch = [x for x in singles if f'r{i}' in x][0]
        # times = [os.path.getmtime(x) for x in fmatch]
        # min_id = np.argmin(times) # find latest entry
        # fp = fmatch[min_id]
        with open(fmatch, 'rb') as f:
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
    if location.lower() == 'automat':
        result_keys.add('dr_byen')
    res_dict = {k: {k1: v1 for k1, v1 in v.items() if k1 in result_keys} 
                for k, v in results.copy().items()}
    
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
    # if any(any(x == 'dr_byen' for x in y) for y in mro):
    #     mro = [x for x in mro if any(y == 'dr_byen' for y in x)]
    return mro

def _filter_mro(record, mro):
    
    if record.takstsæt == 'th':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_V' not in x]
    elif record.takstsæt == 'tv':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_H' not in x]
    elif record.takstsæt == 'ts':
        mro = [x for x in mro if 'Movia_H' not in x and 'Movia_V' not in x]
    
    return mro


def _join_note(notelist: List[str]) -> str:
    
    return ''.join(
        ('_'.join(j) + r'->' if i!=len(notelist)-1 else '_'.join(j) 
         for i, j in enumerate(notelist)
         )
        )

def _resolve_tickets(record):
    
    startzone = record.startzone 
    paid_zones = record.betaltezoner
    endzone = record.slutzone
    
    try: 
        startzone = int(startzone)
    except ValueError:
        if not '1003' in startzone:
            startzone = int(startzone[:4])
        else:
            # byen
            pass
    try: 
        endzone = int(endzone)
    except ValueError:
        endzone = int(endzone[:4]) # endzone not needed

 
    paid_zones = int(paid_zones)

    ring_ticket = startzone, paid_zones
    ft_ticket = startzone, endzone
    return ring_ticket, ft_ticket

def _match_single_operator(record, res, mro, trip_threshhold):
    
    
    mro = _filter_mro(record, mro)
    ring_ticket, ft_ticket = _resolve_tickets(record)
    
    if '1001/1003' in ring_ticket or '1001/1003' in ft_ticket:
        mro_dr = [x for x in mro if any(y == 'dr_byen' for y in x)]
        mro_other = [x for x in mro if x not in mro_dr]
        mro = mro_dr + mro_other
        
        ring_ticket = int(ring_ticket[0][:4]), ring_ticket[1]        
        ft_ticket = int(ft_ticket[0][:4]), ft_ticket[1]        
 
    note = []
    for method in mro:
        note.append(method)
        
        
        try:
            d = res[method[0]][method[1]][method[2]].copy()
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
        if r and r['n_trips'] < trip_threshhold:
            continue
        if r is not None:
            break
    else:        
        r = {}   
    
    r['note'] = _join_note(note)
    return r

def _match_single_any(record, res, mro, trip_threshhold):
    
    ring_ticket, ft_ticket = _resolve_tickets(record)

    note = []
    for method in mro:
        note.append(method)
        try:
            d = res[method[0]][method[1]]
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
        if r and r['n_trips'] < trip_threshhold:
            continue
        if r is not None:
            break
    else:        
        r = {}   
    
    r['note'] = _join_note(note)
    return r


def _method_resolution_any(res, length):
    
    mro = []
    paid = []
    
    for k, v in res.items():
        if length == 'short':
            start_any = (k, 'short_ring')
            mro.append(start_any)           
        if length == 'long':
            start_any = (k, 'long')  
            mro.append(start_any)                         
            start_any_ring = (k, 'long_ring')
            mro.append(start_any_ring)
            
        start_any_paid = (k, 'paid_zones')
        paid.append(start_any_paid)

    if length == 'long':
        mro = sorted(mro, key=itemgetter(1, 0))
    mro = mro + paid

    return mro

def _result_frame(rdict, frame):

    out_frame = pd.DataFrame.from_dict(rdict).T
    out_frame.index.name = 'NR'
    out_frame = out_frame.reset_index()
    
    test_out = pd.merge(frame, out_frame, on='NR', how='left')
    test_out.note = test_out.note.fillna('')
    test_out = test_out.fillna(0)
    
    main_cols = list(frame.columns)
    new = [x for x in test_out if x not in main_cols]
    ops = [x for x in new if x not in ('note', 'n_trips')]
    out_cols = main_cols + sorted(ops) + ['n_trips', 'note']
    test_out = test_out[out_cols]
    
    return test_out
    
    
def _any_single_merge(sales_idxs, location_sales, data, single_results, min_trips, loc):
    
    sales_nr = set(sales_idxs['enkeltbillet'] + sales_idxs['lang enkeltbillet'])
    location_nr = set(chain(*location_sales.values()))
    
    if loc:
        wanted = sales_nr - location_nr
    else:
        wanted = sales_nr
    sub_data = data.query("NR in @wanted")
    sub_tuples = list(sub_data.itertuples(index=False, name='Sale'))
    
    res = {k: v['all'] for k, v in single_results.copy().items()}

    mro_short = _method_resolution_any(res, 'short')
    mro_long = _method_resolution_any(res, 'long')    

    out = {}
    for record in sub_tuples:
        if record.betaltezoner <= 8:
            out[record.NR] = _match_single_any(
                record, res, mro_short, min_trips)
        else:
            out[record.NR] = _match_single_any(record, res, mro_long, min_trips)

             
    outframe = _result_frame(out, sub_data)   

    return outframe

def _location_merge(location_sales, data, single_results, min_trips):
    
    final = []
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
                out[record.NR] = _match_single_operator(
                    record, res, mro_short, min_trips
                    )
            else:
                out[record.NR] = _match_single_operator(
                    record, res, mro_long, min_trips
                    )
        outframe = _result_frame(out, sub_data)           
        final.append(outframe)
               
    return pd.concat(final)


def _single_tickets(sales_idxs, 
                    location_sales, 
                    data, 
                    single_results, 
                    min_trips, 
                    loc=False):
    
    any_start = _any_single_merge(
        sales_idxs, location_sales, data, single_results, min_trips,
        loc=loc)
    
    if loc:
        specific_start = _location_merge(
            location_sales, data, single_results, min_trips
            )
        
        out = pd.concat([any_start, specific_start])
    else:
        out = any_start

    return out.sort_values('NR')


# =============================================================================
# Pendler tickets
# =============================================================================

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

def _load_kombi_map_shares(year: int, model: int) -> pd.core.frame.DataFrame:

    fp = os.path.join(
        THIS_DIR, '__result_cache__', 
        f'{year}', 'pendler', 
        f'zonerelations{year}_model_{model}.csv'
        )
    df = pd.read_csv(
        fp, 
        dtype={'PaidZones': int}, 
        usecols = ['StartZone', 'DestinationZone', 'PaidZones', 
                   'movia', 'stog', 'first', 'metro', 'dsb', 
                   'n_users', 'n_period_cards', 'n_trips']
        )
    df = df.set_index(['StartZone', 'DestinationZone', 'PaidZones'])
    d = df.to_dict(orient='index')

    return d

def _load_nzone_shares(year: int, model: int):

    
    takst_map = {
        'dsb': 'dsb', 
        'th': 'hovedstad', 
        'ts': 'sydsjælland', 
        'tv': 'vestsjælland'
        }
    
    filedir = os.path.join(THIS_DIR, '__result_cache__', f'{year}', 'pendler')
    files = glob.glob(os.path.join(filedir, '*.csv'))
    kombi = [x for x in files if 'kombi_paid_zones' in x and f'model_{model}' in x]


    out = {}
    for file in kombi:
        
        df = pd.read_csv(file, index_col=0)
        d = df.to_dict(orient='index')
        for k, v in takst_map.items():
            if v in file:
                out[k] = d
                break
               
    return out


def _kombimatch(valgt, kombi_results):
    
    return kombi_results.get(valgt, {})
    
  
def _kombi_paid():
    
    return 


def _join_note_p(notelist: List[str]) -> str:
    
    return ''.join(
        (''.join(j) + r'->' if i!=len(notelist)-1 else ''.join(j) 
         for i, j in enumerate(notelist)
         )
        )


def _match_pendler_record(
        record, kombi_results, 
        zone_relation_results, 
        paid_zones_results, 
        min_trips
        ):
    
    valgt = ast.literal_eval(record.valgtezoner)
    takst = record.takstsæt
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
    if valgt:  
        if 1002 in valgt and 1001 not in valgt:
            note.append('INVALID_KOMBI')
        mro = ['kombimatch', f'kombi_paid_zones_{takst}']       
        for method in mro:           
            note.append(method)
            r = kombi_results.get(valgt, {})
            if r and r['n_trips'] >= min_trips:
                break
            else:
                r = paid_zones_results[takst].get(paid, {})
            
    else:
        mro = ['zonerelation_match', f'kombi_paid_zones_{takst}']
        for method in mro:
            note.append(method)            
            r = zone_relation_results.get((start, end, paid), {})
            if r and r['n_trips'] >= min_trips:
                break
            else:
                r = paid_zones_results[takst].get(paid, {})

   
    if r['n_trips'] < min_trips:
        note.append(f'kombi_paid_all_zones_{takst}')
        r = paid_zones_results[takst].get(99, {})
    
    out = r.copy()
    out['note'] = _join_note_p(note)    
    return out
   
def _get_zone_relation_results(kombi_results):
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
    
    
def _pendler_tickets(
        sales_idxs: Dict,
        data: pd.core.frame.DataFrame,
        year: int, 
        min_trips: int,
        model: int
        ) -> pd.core.frame.DataFrame:

    # TODO - set this in config
    pendler_ids = \
        sales_idxs.get('pendlerkort', ()) + \
            sales_idxs.get('erhvervskort', ())  + \
                sales_idxs.get('virksomhedskort', ())  + \
                    sales_idxs.get('pensionistkort', ())  + \
                        sales_idxs.get('ungdomskort xu', ())  + \
                            sales_idxs.get('flexcard 7 dage', ()) + \
                                sales_idxs.get('flexcard', ()) + \
                                    sales_idxs.get('ungdomskort vu', ())  + \
                                        sales_idxs.get('ungdomskort uu', ())

    sub_data = data.query("NR in @pendler_ids")
    sub_tuples = list(sub_data.itertuples(index=False, name='Sale'))
    
    kombi_results = _load_kombi_shares(year, model)    
    zone_relation_results = _get_zone_relation_results(kombi_results)
    paid_zones_results = _load_nzone_shares(year, model)

       
    bad = set()
    out = {}
    for record in sub_tuples:
        try:
            out[record.NR] = _match_pendler_record(
                record, kombi_results, 
                zone_relation_results, 
                paid_zones_results, 
                min_trips
                )                
        except:
            bad.add(record.NR)
    # merge into _result_frame func
    out_frame = pd.DataFrame.from_dict(out).T
    out_frame.index.name = 'NR'
    out_frame = out_frame.reset_index()
    
    test_out = pd.merge(sub_data, out_frame, on='NR', how='left')
    test_out.note = test_out.note.fillna('')
    test_out = test_out.fillna(0)
    
    main_cols = list(data.columns)
    new = [x for x in test_out if x not in main_cols]
    ops = [x for x in new if x not in ('note', 'n_trips', 'n_users', 'n_period_cards')]
    out_cols = main_cols + sorted(ops) + ['n_trips', 'n_period_cards', 'n_users', 'note']
    test_out = test_out[out_cols]
    
    return test_out


    
def _load_other_results(year: int, model: int) -> Dict:

    fp = os.path.join(
        THIS_DIR, '__result_cache__', 
        f'{year}', 'preprocessed', 
        f'subtakst_model_{model}.pickle'
        )
    
    with open(fp, 'rb') as f:
        d = pickle.load(f)
    return d
    

def _match_other_record(
        record, other_results, 
        small_ids, big_ids
        ):
    takst = record.takstsæt
    product = record.produktnavn
    
    
    note = None
    if record.NR in small_ids:
        r =  other_results['city']
        note = 'all_inner_city_rabat0'
    elif record.NR in big_ids:
        if 'city' in product or 'copenhagen' in product:
            r = other_results['th']
            note = 'all_th_rabat0'
        else:
            r = other_results[f'{takst}']
            note = f'all_{takst}_rabat0'
    
    out = r.copy()
    out['note'] = note
    return out

def _other_tickets(
        sales_idxs: Dict,
        data: pd.core.frame.DataFrame,
        year: int, 
        model: int
        ):

    results = _load_other_results(year, model)
    results = results['alltrips']

    
    small_tickets = {
        'city pass small',
        'citypass small 24 timer',
        'citypass small 120 timer',
        'citypass small 48 timer',
        'citypass small 72 timer',
        'citypass small 96 timer',
        'mobilklippekort'
        }

    big_tickets = {
        'institutionskort, 20 børn', 'travel pass',
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
        'enkeltbillet refusion'
        }
      
    small_ids = set().union(*[set(sales_idxs.get(x, set())) for x in small_tickets])
    big_ids = set().union(*[set(sales_idxs.get(x, set())) for x in big_tickets])
    
    sub_data = data.query("NR in @small_ids or NR in @big_ids")
    
    records = list(sub_data.itertuples(index=False, name='Sale'))


    bad = set()
    out = {}
    for record in records:
        try:
            out[record.NR] = _match_other_record(
                record, results, 
                small_ids, 
                big_ids, 
                )                
        except:
            bad.add(record.NR)
    
    out_frame = _result_frame(out, sub_data)
    
    return out_frame

def single_ticket(sales_idxs, location_sales, data, single_results):
    
    from collections import Counter
    for i in [1, 2, 5, 10, 20, 50]:
        single_output = _single_tickets(
            sales_idxs, location_sales, data, single_results, i
            )           
        single_output.to_csv(f"H:/single_ticket_mintrips{i}.csv", index=False)           

        n = len(single_output)
        counts = Counter(single_output.note.str.count('->'))
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])    
        counts = {ordinal(k+1) + ' try': (v / n) * 100 for k, v in counts.items()}
        stats = pd.DataFrame.from_dict(counts, orient='index')
        stats = stats.reset_index()
        stats.columns = ['attempts', 'percentage']
        stats.to_csv(f"H:/single_ticket_mintrips{i}_stats.csv", index=False)

def main():
    
    parser = TableArgParser('year', 'model')
    
    args = parser.parse()
    year = args['year']
    model = args['model']
    
    data = _load_sales_data(year)
    sales_idxs = _sales_ref(data)
    location_idxs = _location_ref(data)
    location_sales = _get_location_sales(location_idxs, sales_idxs)

    if model == 3:
        single_model = 1
    else:
        single_model = model
    
    single_results = _get_single_results(year, single_model)
    
    minimum_trips = 1
    
    single_output = _single_tickets(
        sales_idxs, location_sales, data, 
        single_results, minimum_trips,
        loc=USE_LOCATIONS
        )
    
    pendler_output = _pendler_tickets(
        sales_idxs, data, year, minimum_trips, model
        )
      
    if model == 3:
        other_model = 1
    else:
        other_model = model        
    
    other_output = _other_tickets(sales_idxs, data, year, other_model)
 
    output = pd.concat([single_output, pendler_output, other_output])
    output = output.sort_values('NR')
    output = output.fillna(0)

    
    cols = ['dsb', 'first', 'stog', 'movia', 'metro']
    
    for col in cols:
        output.loc[:, f'{col}_Andel'] = \
            output.loc[:, 'omsætning'] * output.loc[:, col]

    output = output[
        ['NR', 'salgsvirksomhed', 'indtægtsgruppe', 
         'salgsår', 'salgsmåned', 'takstsæt', 
         'produktgruppe', 'produktnavn', 'kundetype', 
         'salgsmedie', 'betaltezoner', 'startzone', 
         'slutzone', 'valgtezoner', 'omsætning',
         'antal', 'dsb', 'first', 'stog',  'movia', 
         'metro', 'n_trips', 'note', 'n_period_cards', 
         'n_users', 'dsb_Andel', 'first_Andel', 
         'stog_Andel', 'movia_Andel', 'metro_Andel']
        ]
    
    output.to_csv(f'takst_sjælland{year}_model_{model}.csv', index=False)

       
if __name__ == "__main__":
      main()

