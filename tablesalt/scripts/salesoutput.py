# -*- coding: utf-8 -*-
"""
This is the last script to run to create output for the "kildefordeling"
or revenue sharing between operators


What does it do?
================

This script joins the merged sales input data with the aggregated results
sets produced by singlekeys.py, pendlerkeys.py and subtakst.py

Each record in the sales data gets assigned an operator share based on
the settings in the configuration file.

USAGE
=====
To merge all sales with all results for 2019 using model 1
    python ./path/to/scripts/salesoutput.py -y 2019 -m 1

Configuration File
==================

The file salesoutconfig.json contains the settings for how you wish to
match a product sales record with the aggregate data.

Below are the default settings for the merging rules

start_locations
---------------
Here you place the "salgsmedie" that indicates which operator sells the ticket
and thus, theoretically, which operator services the first leg of the trip

single_tickets
--------------
A list of the product names to be merged using single ticket aggregations

pendler_tickets
---------------
A list of product names to be merged using pendler ticket aggregations

city_small
----------
A list of product names to merge using the small city pass (zones 1001, 1002, 1003, 1004)
results from the subtakst script

city_large
----------
A list of product names to merge using the large city pass (zones 1001-1099)
results from the subtakst script

use_locations
-------------
a boolean value (true/false). Whether to enforce the start_locations.
That is, if a sales record as a starting zone 1001 and salgsmedie "automat"
whether to aggregate only trips that start on the metro or not.
The default is false

minimum_trips
-------------
The minimum number of trips required to merge a certain aggregation before
using a fallback method
The default is 1

{
    "start_locations": {
        "automat": "metro",
        "nautila": "dsb",
        "lokaltog-automater i 25 nordsjællandske togsæt": "movia",
        "enkeltbilletter bus": "movia"
    },
    "single_tickets": [
        "enkeltbillet",
        "lang enkeltbillet",
        "forsorgsbilletter"
    ],
    "pendler_tickets": [
        "pendlerkort",
        "erhvervskort",
        "virksomhedskort",
        "pensionistkort",
        "ungdomskort xu",
        "flexcard 7 dage",
        "flexcard",
        "ungdomskort vu",
        "ungdomskort uu",
        "skolekort"
    ],
    "city_small": [
        "city pass small",
        "citypass small 24 timer",
        "citypass small 120 timer",
        "citypass small 48 timer",
        "citypass small 72 timer",
        "citypass small 96 timer",
        "mobilklippekort",
        "park and ride"
    ],
    "city_large": [
        "institutionskort, 20 børn",
        "travel pass",
        "kulturnatten",
        "citypass large 120 timer",
        "citypass large 24 timer",
        "børnealderskompensation",
        "city pass large",
        "citypass large 96 timer",
        "citypass large 72 timer",
        "copenhagen card",
        "citypass large 48 timer",
        "skoleklassekort",
        "institutionskort, 15 børn",
        "blindekort",
        "off peak-kompensation",
        "mobilklippekort",
        "turistbillet",
        "print-selv-billet",
        "enkeltbillet refusion",
        "ungdomskort uu - fritid",
        "ungdomskort uu - kompensation",
        "ungdomskort vu - fritid",
        "ungdomskort vu - kompensation",
        "ungdomskort xu - fritid",
        "dsb-salg uspecificeret",
        "bornholmtrafikken"
    ],
    "use_locations": false,
    "minimum_trips": 1
}

"""
import ast
import glob
import json
import os
import pickle
import pkg_resources
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TypedDict, Union

import msgpack #type: ignore
import pandas as pd #type: ignore

from tablesalt.preprocessing.parsing import TableArgParser #type: ignore

THIS_DIR = Path(__file__).parent

def _load_outconfig():
    """load the configuration for sales output"""
    fp = THIS_DIR / 'salesoutconfig.json'
    with open(fp, 'r') as f:
        config = json.load(f)
    return config

CONFIG = _load_outconfig()
LOCATIONS = CONFIG['start_locations']
SINGLE_IDS = CONFIG['single_tickets']
PENDLER_IDS = CONFIG['pendler_tickets']
CITY_SMALL = CONFIG['city_small']
CITY_LARGE = CONFIG['city_large']
USE_LOCATIONS = CONFIG['use_locations']
MINIMUM_NTRIPS = CONFIG['minimum_trips']

RESULT_MAP = {
    'dsb': ('D**', 'D*', 'D'),
    'metro': ('Metro',),
    'movia': ('Movia_H', 'Movia_S', 'Movia_V')
    }

# for type checking
class ShareDict(TypedDict, total=False):
    dsb: float
    first: float
    movia: float
    stog: float
    n_trips: int
    metro: float

class RegionDict(TypedDict, total=False):
    tv: ShareDict
    ts: ShareDict
    dsb: ShareDict
    th: ShareDict

STICKET_DICT = Dict[Tuple[int, int], ShareDict]
PAID_DICT = Dict[int, ShareDict]
SINGLE_DICT = Dict[str, Dict[str, Union[RegionDict, STICKET_DICT, PAID_DICT]]]

def _proc_sales(frame: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """process the sales dataframe"""

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].apply(
        lambda x: tuple(sorted(ast.literal_eval(x)))
        )

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].astype(str)

    frame.loc[:, 'slutzone'] = frame.loc[:, 'slutzone'].fillna('1000')
    frame.loc[:, 'startzone'] = frame.loc[:, 'startzone'].fillna('1000')

    return frame


def _load_sales_data(year: int) -> pd.core.frame.DataFrame:
    """load and process the merged sales data csv"""

    fp = (THIS_DIR / '__result_cache__' /
          f'{year}' / 'preprocessed' /
          'mergedsales.csv')

    df = pd.read_csv(fp, low_memory=False)

    return _proc_sales(df)

def _get_single_results(year: int,  model: int) -> SINGLE_DICT:
    """load the single tickets results from the result cache"""
    fp = THIS_DIR / '__result_cache__'/ f'{year}' / 'preprocessed'

    singles = list(fp.glob('single_results*'))
    singles = [x for x in singles if f'model_{model}' in x.name]
    out = {}
    for i in range(3): # using r0, r1, r2
        fmatch = [x for x in singles if f'r{i}' in x.name][0]
        with open(fmatch, 'rb') as f:
            res = pickle.load(f)
        out[f'rabat{i}'] = res

    return out

IDXS = Dict[str, Tuple[int, ...]]

def _sales_ref(frame: pd.core.frame.DataFrame) -> IDXS:
    """get the sales NR values for each product name"""

    products = set(frame.loc[:, 'produktnavn'])
    sales_idxs = {}

    for prod in products:
        sales_idxs[prod] = tuple(
            frame.query("produktnavn == @prod").loc[:, 'NR']
            )

    return sales_idxs

def _location_ref(
        frame: pd.core.frame.DataFrame
        ) -> IDXS:
    """get the sales NR values for each type of sales equipment"""

    sources = set(frame.loc[:, 'salgsmedie'])
    location_idxs = {}

    for src in sources:
        location_idxs[src] = tuple(
            frame.query("salgsmedie == @src").loc[:, 'NR']
            )

    return location_idxs


def _get_location_sales(
    location_idxs: IDXS,
    sales_idxs: IDXS
    ) -> Dict[str, Set[int]]:
    """get the sales NR for each sales type of sales equipment designated
    as a starting location"""

    wanted_start_locations = {
        k: v for k, v in location_idxs.items() if k in LOCATIONS
        }

    single = set()
    for name in SINGLE_IDS:
        try:
            single.update(set(sales_idxs[name]))
        except KeyError:
            continue
    location_sales = {}
    for k, v in wanted_start_locations.items():
        location_sales[k] = set(v).intersection(single)

    return location_sales

def _get_location_results(
    location: str,
    results: SINGLE_DICT
    ) -> SINGLE_DICT:
    """subset the results for the given sales location"""

    operator = LOCATIONS[location]
    result_keys = set(RESULT_MAP[operator])
    result_keys.add('all')
    if location.lower() == 'automat':
        result_keys.add('dr_byen')
    res_dict = {k: {k1: v1 for k1, v1 in v.items() if k1 in result_keys}
                for k, v in results.copy().items()}

    return res_dict

def _method_resolution_operator(
    results: SINGLE_DICT,
    ticket_length: str
    ) -> List[Tuple[str, str, str]]:
    """create the method resolution order for
    single tickets with starting operator reqquirement"""

    mro = []
    any_start = []
    paid = []
    paid_any = []

    for k, v in results.items():
        if ticket_length == 'short':
            start_op = [(k, x, 'short_ring') for x in v if x != 'all']
            start_any = [(k, x, 'short_ring')for x in v if x == 'all']

        if ticket_length == 'long':
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

    if ticket_length == 'long':
        mro = sorted(mro, key=itemgetter(1, 0))
    mro = mro + any_start + paid + paid_any
    # if any(any(x == 'dr_byen' for x in y) for y in mro):
    #     mro = [x for x in mro if any(y == 'dr_byen' for y in x)]
    return mro

def _filter_mro(
    record: Tuple[Any, ...],
    mro: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
    """filter out the specific takst starting operators for movia"""
    # This may change, so look out for this
    if record.takstsæt == 'th':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_V' not in x]
    elif record.takstsæt == 'tv':
        mro = [x for x in mro if 'Movia_S' not in x and 'Movia_H' not in x]
    elif record.takstsæt == 'ts':
        mro = [x for x in mro if 'Movia_H' not in x and 'Movia_V' not in x]

    return mro


def _join_note(notelist: List[str]) -> str:
    """join the list of notes to form a string"""
    return ''.join(
        ('_'.join(j) + r'->' if i!=len(notelist)-1 else '_'.join(j)
         for i, j in enumerate(notelist))
        )

def _resolve_tickets(
    record: Tuple[Any, ...]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """get the start, end and paid zones of a sales record
    """
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
        endzone = 0 # endzone not needed for single ring or NaN

    paid_zones = int(paid_zones)

    ring_ticket = startzone, paid_zones
    from_to_ticket = startzone, endzone
    return ring_ticket, from_to_ticket

def _match_single_operator(
    record: Tuple[Any, ...],
    res: SINGLE_DICT,
    mro: List[Tuple[str, str, str]],
    trip_threshhold: int
    ):
    """match a sales record to a single result for starting operator locations

    :param record: itertuple from dataframe
    :type record: Tuple[Any, ...]
    :param res: the single result dictionary
    :type res: SINGLE_DICT
    :param mro: the method resolution order
    :type mro: List[Tuple[str, str, str]]
    :param trip_threshhold: minimum number or sample trips allowed
    :type trip_threshhold: int
    :return: a matched result
    :rtype: [type]
    """

    mro = _filter_mro(record, mro)
    ring_ticket, from_to_ticket = _resolve_tickets(record)

    if '1001/1003' in ring_ticket or '1001/1003' in from_to_ticket:
        mro_dr = [x for x in mro if any(y == 'dr_byen' for y in x)]
        mro_other = [x for x in mro if x not in mro_dr]
        mro = mro_dr + mro_other

        ring_ticket = int(ring_ticket[0][:4]), ring_ticket[1]
        from_to_ticket = int(from_to_ticket[0][:4]), from_to_ticket[1]

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
            r = d.get(from_to_ticket, {})
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

def _match_single_any(
    record: Tuple[Any, ...],
    res: SINGLE_DICT,
    mro: List[Tuple[str, str, str]],
    trip_threshhold: int
    ):
    """match a sales record to a single result if starting operators
    are not required

    :param record: itertuple from dataframe
    :type record: Tuple[Any, ...]
    :param res: the single result dictionary
    :type res: SINGLE_DICT
    :param mro: the method resolution order
    :type mro: List[Tuple[str, str, str]]
    :param trip_threshhold: minimum number or sample trips allowed
    :type trip_threshhold: int
    :return: a matched result
    :rtype: [type]
    """

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
    """
    create the merging method resolution for results that
    don't use locations
    """

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
    """create a dataframe from result dict and a sales dataframe
    """

    out_frame = pd.DataFrame.from_dict(rdict).T
    out_frame.index.name = 'NR'
    out_frame = out_frame.reset_index()

    output = pd.merge(frame, out_frame, on='NR', how='left')
    output.note = output.note.fillna('')
    output = output.fillna(0)

    main_cols = list(frame.columns)
    new = [x for x in output if x not in main_cols]
    ops = [x for x in new if x not in ('note', 'n_trips')]
    out_cols = main_cols + sorted(ops) + ['n_trips', 'note']
    output = output[out_cols]

    return output


def _any_single_merge(
        sales_idxs: IDXS,
        location_sales: Dict[str, Set[int]],
        data: pd.core.frame.DataFrame,
        single_results: SINGLE_DICT,
        min_trips: int,
        loc: bool
        ) -> pd.core.frame.DataFrame:
    """merge the sales data of single tickets with the matched results

    :param sales_idxs: a set of sales nr from the sales data
    :type sales_idxs: IDXS
    :param location_sales: the sales numbers of the start locations
    :type location_sales: Dict[str, Set[int]]
    :param data: the sales data
    :type data: pd.core.frame.DataFrame
    :param single_results: the single ticket result dictionary
    :type single_results: SINGLE_DICT
    :param min_trips: the minimum number of samples
    :type min_trips: INT
    :param loc: to use starting locations or not
    :type loc: BOOL
    :return: merged single sales results
    :rtype: pd.core.frame.DataFrame
    """
    sales_nr = set()

    for name in SINGLE_IDS:
        try:
            sales_nr.update(set(sales_idxs[name]))
        except KeyError:
            continue

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
                record, res, mro_short, min_trips
                )
        else:
            out[record.NR] = _match_single_any(
                record, res, mro_long, min_trips
                )


    outframe = _result_frame(out, sub_data)

    return outframe

def _location_merge(
    location_sales: Dict[str, Set[int]],
    data: pd.core.frame.DataFrame,
    single_results: SINGLE_DICT,
    min_trips: int
    ) -> pd.core.frame.DataFrame:
    """merge the sales data and single results based on starting location

    :param location_sales: sales numbers for each starting location
    :type location_sales: Dict[str, Set[int]]
    :param data: the sales data matching sales nrs for locations
    :type data: pd.core.frame.DataFrame
    :param single_results: the single ticket result dictionary
    :type single_results: SINGLE_DICT
    :param min_trips: the minimum number of samples
    :type min_trips: int
    :return: merged single sales results
    :rtype: pd.core.frame.DataFrame
    """

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


def _single_tickets(
    sales_idxs: IDXS,
    location_sales: Dict[str, Set[int]],
    data: pd.core.frame.DataFrame,
    single_results: SINGLE_DICT,
    min_trips: int,
    loc=False
    ) -> pd.core.frame.DataFrame:
    """merge the sales data and the single ticket results

    :param sales_idxs: a set of sales nr from the sales data
    :type sales_idxs: IDXS
    :param location_sales: the sales numbers of the start locations
    :type location_sales: Dict[str, Set[int]]
    :param data: the sales data matching sales nrs for single tickets
    :type data: pd.core.frame.DataFrame
    :param single_results: the single ticket result dictionary
    :type single_results: SINGLE_DICT
    :param min_trips: the minimum number of samples
    :type min_trips: int
    :param loc: whether to use the starting location or not, defaults to False
    :type loc: bool, optional
    :return: merged single sales results
    :rtype: pd.core.frame.DataFrame
    """

    any_start = _any_single_merge(
        sales_idxs, location_sales,
        data, single_results,
        min_trips, loc=loc
        )

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

def _load_kombi_shares(
    year: int,
    model: int
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
    """load the results for the chosen zones for kombi users

    :param year: analysis year
    :type year: int
    :param model: the model number
    :type model: int
    :return: result dictionary for each zone combination
    :rtype: dict
    """

    fp =  (THIS_DIR / '__result_cache__' / f'{year}' /
           'pendler' / f'pendlerchosenzones{year}_model_{model}.csv')

    df = pd.read_csv(fp, index_col=0)
    df.rename(columns={'S-tog': 'stog'}, inplace=True)

    d = df.to_dict(orient='index')

    return {ast.literal_eval(k): v for k, v in d.items()}


def _unpack_valid_zones(zonelist: List[Dict[str, int]]) -> Tuple[int, ...]:
    """return a tuple of zones from the relations from DOT API

    :param zonelist: the list of dicts
    :type zonelist: [type]
    :return: a tuple of zone numbers
    :rtype: Tuple[int, ...]
    """

    return tuple(x['ZoneID'] for x in zonelist)

def _proc_zone_relations(zone_rels: dict):
    """process the loaded zone_relation

    :param zone_rels: [description]
    :type zone_rels: dict
    :return: [description]
    :rtype: [type]
    """

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
    load the the zone_relations keys produced from the DOT API
    """

    fp = pkg_resources.resource_filename(
        'tablesalt',
        '/resources/revenue/zone_relations.msgpack'
        )

    with open(fp, 'rb') as f:
        zone_relations = msgpack.load(f, strict_map_key=False)


    return _proc_zone_relations(zone_relations)

def _load_kombi_map_shares(
    year: int,
    model: int
    ) -> Dict:
    """load the results of the kombi shares that are aggregated from, to, paid
    for the given model number
    """
    fp = (THIS_DIR / '__result_cache__' / f'{year}' /
          'pendler' / f'zonerelations{year}_model_{model}.csv')

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

def _load_nzone_shares(
    year: int,
    model: int
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
    """load the results for the results aggregated by paid zones

    :param year: the year of analysis
    :type year: int
    :param model: the model number
    :type model: int
    :return: dictionary of results
    :rtype: Dict[str, Dict[int, Dict[str, float]]]
    """


    takst_map = {
        'dsb': 'dsb',
        'th': 'hovedstad',
        'ts': 'sydsjælland',
        'tv': 'vestsjælland'
        }

    filedir = THIS_DIR /'__result_cache__' / f'{year}' / 'pendler'

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


def _kombimatch(valgt, kombi_results):

    return kombi_results.get(valgt, {})

def _join_note_p(notelist: List[str]) -> str:
    """join the list for the pendler mro

    :param notelist: a list of the notes
    :type notelist: List[str]
    :return: a string of the methods tried
    :rtype: str
    """

    return ''.join(
        (''.join(j) + r'->' if i!=len(notelist)-1 else ''.join(j)
         for i, j in enumerate(notelist)
         )
        )

def _get_closest_kombi(chosen_zones):
    """just take the chosen zones tuple and add zone 1001
    """
    # only for zone 1001/1002 problem right now
    new_chosen_zones = [1001] + list(chosen_zones)
    return tuple(new_chosen_zones)

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


def _pendler_tickets(
        sales_idxs: IDXS,
        data: pd.core.frame.DataFrame,
        year: int,
        min_trips: int,
        model: int
        ) -> pd.core.frame.DataFrame:
    """match the sales data for pendler tickets to results

    :param sales_idxs: a set of sales nr from the sales data
    :type sales_idxs: IDXS
    :param data: the sales data
    :type data: pd.core.frame.DataFrame
    :param year: the year of analysis
    :type year: int
    :param min_trips: the minimum number of samples
    :type min_trips: int
    :param model: the model number
    :type model: int
    :return: merged sales and results
    :rtype: pd.core.frame.DataFrame
    """

    pendler_ids = set()

    for name in PENDLER_IDS:
        try:
            pendler_ids.update(set(sales_idxs[name]))
        except KeyError:
            continue

    sub_data = data.query("NR in @pendler_ids")
    sub_tuples = list(sub_data.itertuples(index=False, name='Sale'))

    kombi_results = _load_kombi_shares(year, model)
    zone_relation_results = _load_zone_relation_results(kombi_results)
    paid_zones_results = _load_nzone_shares(year, model)

    bad = set()
    out = {}
    for record in sub_tuples:
        try:
            result =  _match_pendler_record(
                record, kombi_results,
                zone_relation_results,
                paid_zones_results,
                min_trips
                )
            # if 'INVALID_KOMBI' in result['note']:
            #     break
            out[record.NR] = result
        except:
            bad.add(record.NR)
    # merge into _result_frame func
    out_frame = pd.DataFrame.from_dict(out).T
    out_frame.index.name = 'NR'
    out_frame = out_frame.reset_index()

    output = pd.merge(sub_data, out_frame, on='NR', how='left')
    output.note = output.note.fillna('')
    output = output.fillna(0)

    main_cols = list(data.columns)
    new = [x for x in output if x not in main_cols]
    ops = [x for x in new if x not in ('note', 'n_trips', 'n_users', 'n_period_cards')]
    out_cols = main_cols + sorted(ops) + ['n_trips', 'n_period_cards', 'n_users', 'note']
    output = output[out_cols]

    return output



def _load_other_results(year: int, model: int) -> Dict:
    """load results for paid zones in the separate takstzones in sjælland

    :param year: the analysis year
    :type year: int
    :param model: the model number
    :type model: int
    :return: the result dictionary
    :rtype: Dict
    """

    fp = os.path.join(
        THIS_DIR, '__result_cache__',
        f'{year}', 'preprocessed',
        f'subtakst_model_{model}.pickle'
        )

    with open(fp, 'rb') as f:
        d = pickle.load(f)
    return d


def _match_other_record(
        record: Tuple[Any, ...],
        other_results,
        small_ids: Set[int],
        big_ids: Set[int]
        ):
    """match a sales record to a result for tickets not pendler or single

    :param record: sales record from itertuples
    :type record: Tuple[Any, ...]
    :param other_results: result dictionary for other ticket types
    :type other_results: dict
    :param small_ids: sales ids for citypass small
    :type small_ids: Set[int]
    :param big_ids: sales ids for citypass large
    :type big_ids: Set[int]
    :return: result dict for record
    :rtype: [type]
    """
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
        sales_idxs: IDXS,
        data: pd.core.frame.DataFrame,
        year: int,
        model: int
        ) -> pd.core.frame.DataFrame:
    """create the output for the tickets in the citypass small and large
    categories

    :param sales_idxs: the dictionary of sales indices
    :type sales_idxs: IDXS
    :param data: the sales data for the nr matching other tickets
    :type data: pd.core.frame.DataFrame
    :param year: the year of analysis
    :type year: int
    :param model: the number of the model
    :type model: int
    :return: a dataframe of sales matched with results
    :rtype: pd.core.frame.DataFrame
    """
    results = _load_other_results(year, model)
    results = results['alltrips']
    small_ids: Set[int]
    big_ids: Set[int]
    small_ids = set().union(*[set(sales_idxs.get(x, set())) for x in CITY_SMALL])
    big_ids = set().union(*[set(sales_idxs.get(x, set())) for x in CITY_LARGE])

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

def single_ticket(
    sales_idxs: IDXS,
    location_sales: Dict[str, Set[int]],
    data: pd.core.frame.DataFrame,
    single_results: SINGLE_DICT
    ):
    """function that gets results for different minimum samples

    :param sales_idxs: the dictionary of sales indices
    :type sales_idxs: [type]
    :param location_sales: the sales numbers of the start locations
    :type location_sales: Dict[str, Set[int]]
    :param data: the sales data
    :type data: pd.core.frame.DataFrame
    :param single_results: result dict for single
    :type single_results: SINGLE_DICT
    """

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
def main(year: int, model: int) -> None:
    """create the merged sales and results csv file

    :param year: the year of analysis
    :type year: int
    :param model: the desired model results 1/2/3
    :type model: int
    """

    data = _load_sales_data(year) # merged sales data. run salesdatamerge.py
    sales_idxs = _sales_ref(data)
    location_idxs = _location_ref(data)
    location_sales = _get_location_sales(location_idxs, sales_idxs)


    single_results = _get_single_results(year, model)

    single_output = _single_tickets(
        sales_idxs,
        location_sales,
        data,
        single_results,
        MINIMUM_NTRIPS,
        loc=USE_LOCATIONS
        )

    pendler_output = _pendler_tickets(
        sales_idxs,
        data,
        year,
        MINIMUM_NTRIPS,
        model
        )

    other_output = _other_tickets(
        sales_idxs,
        data,
        year,
        model
        )

    output = pd.concat([single_output, pendler_output, other_output])
    output = output.sort_values('NR')
    output = output.fillna(0)
    initial_columns = list(data.columns)

    stats_columns = ['n_trips', 'n_users', 'n_period_cards', 'note']
    operator_columns = [
        x for x in output.columns if x not in initial_columns
        and x not in stats_columns
        ]

    andel_columns = []
    for col in operator_columns:
        new_col = f'{col}_andel'
        andel_columns.append(new_col)
        output.loc[:, f'{col}_andel'] = \
            output.loc[:, 'omsætning'] * output.loc[:, col]

    col_order = initial_columns + operator_columns + andel_columns + stats_columns
    output = output[col_order]
    output = add_city_note(output)
    fp = (THIS_DIR / '__result_cache__' / f'{year}'/
          f'takst_sjælland{year}_model_{model}.csv')

    output.to_csv(fp, index=False)


if __name__ == "__main__":

    parser = TableArgParser('year', 'model')
    args = parser.parse()
    year = args['year']
    model = args['model']
    main(year, model)
