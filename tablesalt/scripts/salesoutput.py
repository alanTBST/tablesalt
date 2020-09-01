# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:18:51 2020

@author: alkj
"""
import ast
import glob
import os

from typing import AnyStr, Dict, List, Tuple, Optional


import pandas as pd


# TODO maybe put this in a config file
LOCATIONS = {
    'automat': 'metro',
    'nautila': 'dsb',
    'lokaltog-automater i 25 nordsjællandske togsæt': 'movia',
    'enkeltbilletter bus': 'movia'
    }


# =============================================================================
# Loading functions
# =============================================================================
def _proc_sales(frame):

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].apply(
        lambda x: tuple(sorted(ast.literal_eval(x)))
        )

    frame.loc[:, 'valgtezoner'] = frame.loc[:, 'valgtezoner'].astype(str)
    return frame

def _load_sales_data(filename: AnyStr) -> pd.core.frame.DataFrame:

    if 'xls' in filename:
        df = pd.read_excel(filename)
        return _proc_sales(df)
    elif 'csv' in filename:
        df = pd.read_csv(filename)
        return _proc_sales(df)

    raise NotImplementedError("unsupported filetype")

def _check_names(frame):

    frame.rename(columns={
        'start_zone': 'startzone',
        'StartZone': 'startzone',
        'DestinationZone': 'slutzone',
        'n_zones': 'betaltezoner',
        'PaidZones': 'betaltezoner',
        'end_zone': 'slutzone',
        'S-tog': 'stog',
        'DSB': 'dsb',
        'Movia_H': 'movia',
        'Metro': 'metro',
        'First': 'first'
        }, inplace=True)
    return frame

def _stringify_merge_cols(frame):

    string_cols = ['betaltezoner', 'slutzone', 'startzone']
    for col in string_cols:      
        try:
            frame.loc[:, 'betaltezoner'] = \
                frame.loc[:, 'betaltezoner'].astype(str)
        except KeyError:
            pass

    return frame

def _load_ringzone_shares() -> pd.core.frame.DataFrame:

    filename = r'sjælland\start_all_short_ring_2019.csv'

    df = pd.read_csv(filename, index_col=0)
    df = _check_names(df)
    df = _stringify_merge_cols(df)

    return df

def _load_long_shares(ring: Optional[bool] = False):

    if not ring:
        filename = r'sjælland\start_all_long_2019.csv'
    else:
        filename = r'sjælland\start_all_long_ring_2019.csv'
    df = pd.read_csv(filename)
    df = _check_names(df)
    df = _stringify_merge_cols(df)

    return df


def _load_operator_shares(
        operator: str,
        length: str,
        ring: Optional[bool] = False
        ) -> pd.core.frame.DataFrame:

    operator = operator.lower()
    length = length.lower()

    filedir = r'sjælland'

    files = glob.glob(os.path.join(filedir, '*.csv'))
    files = [x for x in files if 'start_' in x and 'all' not in x]
    wanted = [x for x in files if operator in x.lower() and length in x.lower()]

    if not ring:
        wanted = [x for x in wanted if 'ring' not in x.lower()]
    else:
        wanted = [x for x in wanted if 'ring' in x.lower()]

    if not wanted:
        raise ValueError(
            f"no files match {operator} and {length} and ring={ring}"
            )

    frames = []
    for f in wanted:
        df = pd.read_csv(f, index_col=0)
        df = _check_names(df)
        if 'Movia_H' in f:
            df = df.query("startzone < 1100")
        elif 'Movia_S' in f:
            df = df.query("startzone > 1200")
        else:
            df = df.query("1100 < startzone < 1200")

        df = _stringify_merge_cols(df)
        frames.append(df)

    return pd.concat(frames)

# =============================================================================
#
# =============================================================================
def _sales_ref(frame: pd.core.frame.DataFrame) -> Dict[str, Tuple[int, ...]]:
    products = set(frame.loc[:, 'produktnavn'])

    sales_numbers = {}

    for prod in products:
        sales_numbers[prod] = tuple(
            frame.query("produktnavn == @prod").loc[:, 'NR']
            )

    return sales_numbers

def _location_ref(
        frame: pd.core.frame.DataFrame
        ) -> Dict[str, Tuple[int, ...]]:

    sources = set(frame.loc[:, 'salgsmedie'])

    sources_numbers = {}

    for src in sources:
        sources_numbers[src] = tuple(
            frame.query("salgsmedie == @src").loc[:, 'NR']
            )

    return sources_numbers


def _add_note(frame, notnull: str, null: str):

    frame['Note'] = ''
    frame.loc[
        frame.loc[:, 'n_trips'].notnull(), 'Note'
        ] = notnull
    frame.loc[
        frame.loc[:, 'n_trips'].isnull(), 'Note'
        ] = null

    return frame


def _extend_note(frame, notnull_ext: str, null_ext: str):

    frame.loc[frame.loc[:, 'n_trips'].notnull(), 'Note'] = \
        frame.loc[frame.loc[:, 'n_trips'].notnull(), 'Note'].apply(
            lambda x: x + fr'/{notnull_ext}'
            )

    frame.loc[frame.loc[:, 'n_trips'].isnull(), 'Note'] = \
        frame.loc[frame.loc[:, 'n_trips'].isnull(), 'Note'].apply(
            lambda x: x + fr'/{null_ext}'
            )

    return frame
# =============================================================================
# Single Tickets
# =============================================================================


def _short_generic(short_frame: pd.core.frame.DataFrame):

    generic_products = short_frame.query("salgsmedie not in @LOCATIONS")
    generic_products = _check_names(generic_products)
    generic_products = _stringify_merge_cols(generic_products)

    generic_results = _load_ringzone_shares()
    generic_results = _stringify_merge_cols(generic_results)

    generic_output = pd.merge(
        generic_products, generic_results,
        on=['startzone', 'betaltezoner'], how='left'
        )

    generic_output = _add_note(
        generic_output, 'short_ringzone', 'short_ringzone/no_trips'
        )

    # fallback allnzones

    return generic_output


def _short_ring_fallback(operator, nullframe):

    short_ring = _load_ringzone_shares()

    nullframe = _stringify_merge_cols(nullframe)

    nullframe = nullframe.drop(
        ['movia', 'first',
         'stog', 'dsb', 'metro',
         'n_trips'], axis=1
        )

    merged_ = pd.merge(
        nullframe, short_ring,
        on=['startzone', 'betaltezoner'],
        how='left'
        )
    merged_ = _extend_note(
        merged_,
        'short_ringzone',
        r'short_ringzone/no_trips'
        )

    return merged_

def _short_specific_operator(short_frame: pd.core.frame.DataFrame):

    start_location = short_frame.query("salgsmedie in @LOCATIONS")

    frames = []
    for k, v in LOCATIONS.items():

        op_shares = _load_operator_shares(v, 'short', ring=True)

        if v == 'metro':
            op_shares = op_shares.query(
                "startzone in ('1001', '1002', '1003', '1004', '1001/1003')"
                )
        op_sales = start_location.query("salgsmedie == @k")

        merged = pd.merge(
            op_sales, op_shares,
            on=['startzone', 'betaltezoner'],
            how='left'
            )

        merged = _add_note(
            merged,
            f'short_ringzone_start_{v}',
            f'short_ringzone_start_{v}/no_trips'
            )
        missed = merged.loc[merged.n_trips.isnull()]

        if not missed.empty:
            ring_fallback = _short_ring_fallback(v, missed)
            merged = pd.concat(
                [merged[merged.n_trips.notnull()], ring_fallback]
                )
            # missed = merged.loc[merged.n_trips.isnull()]
            # if not missed.empty:
            #     nzones_fallback = _all_nzones_fallback()
            #     pass
        frames.append(merged)

    out = pd.concat(frames)

    return out


def _process_short_single(sale_id: Tuple,
                          frame: pd.core.frame.DataFrame,
                          takst: str) -> pd.core.frame.DataFrame:

    sub_frame = frame.query("NR in @sale_id")
    sub_frame = sub_frame.query("takstsæt == @takst")

    short = sub_frame.query("betaltezoner <= 8")
    short = _stringify_merge_cols(short)

    short_gen = _short_generic(short)
    short_operator = _short_specific_operator(short)


    return pd.concat([short_gen, short_operator])



def _long_specific_fallback(operator, nullframe):

    long_ring = _load_operator_shares(
        operator, 'long', ring=True
        )

    nullframe = _stringify_merge_cols(nullframe)

    nullframe = nullframe.drop(
        ['movia', 'first',
         'stog', 'dsb', 'metro',
         'n_trips'], axis=1
        )

    merged_ = pd.merge(
        nullframe, long_ring,
        on=['startzone', 'betaltezoner'],
        how='left'
        )
    merged_ = _extend_note(
        merged_,
        f'long_ringzone_start_{operator}',
        fr'long_ringzone_start_{operator}/no_trips'
        )
    # TODO long, generic_fallback
    # TODO all nzones fallback
    return merged_


def _long_ring_fallback(nullframe):

    long_ring_shares = _load_long_shares(ring=True)

    nullframe = _stringify_merge_cols(nullframe)

    nullframe = nullframe.drop(
        ['movia', 'first',
         'stog', 'dsb', 'metro',
         'n_trips'], axis=1
        )

    merged_ = pd.merge(
        nullframe, long_ring_shares,
        on=['startzone', 'betaltezoner'],
        how='left'
        )
    merged_ = _extend_note(
        merged_,
        'long_ringzone',
        r'long_ringzone/no_trips'
        )

    return merged_

def _all_nzones_fallback(nullframe):

    # TODO write the fucking script
    # take from citypassshares
    return

def _long_specific_operator(long_frame: pd.core.frame.DataFrame):

    start_location = long_frame.query("salgsmedie in @LOCATIONS")

    frames = []
    for k, v in LOCATIONS.items():
        op_shares = _load_operator_shares(v, 'long')
        if v == 'metro':
            op_shares = op_shares.query(
                "startzone in ('1001', '1002', '1003', '1004', '1001/1003')"
                )
        op_sales = start_location.query("salgsmedie == @k")
        merged = pd.merge(
            op_sales, op_shares,
            on=['startzone', 'slutzone'],
            how='left'
            )
        merged = _add_note(
            merged, f'long_start_{v}',
            f'long_start_{v}/no_trips'
            )
        missed = merged.loc[merged.n_trips.isnull()]
        if not missed.empty:
            ring_fallback = _long_specific_fallback(v, missed)
            merged = pd.concat(
                [merged[merged.n_trips.notnull()], ring_fallback]
                )
            # missed = merged.loc[merged.n_trips.isnull()]
            # if not missed.empty:
            #     nzones_fallback = _all_nzones_fallback()
            #     pass

        frames.append(merged)

    out = pd.concat(frames)

    return out

def _long_generic(long_frame: pd.core.frame.DataFrame):
    # long = long_frame.query("betaltezoner > 8")

    generic_products = long_frame.query("salgsmedie not in @LOCATIONS")
    generic_products = _check_names(generic_products)
    generic_products = _stringify_merge_cols(generic_products)

    generic_results = _load_long_shares()

    generic_output = pd.merge(
        generic_products, generic_results,
        on=['startzone', 'slutzone'], how='left'
        )
    generic_output = _add_note(
        generic_output, 'long', 'long/no_trips'
        )
    missed = generic_output[generic_output.n_trips.isnull()]
    if not missed.empty:
        ring_fallback = _long_ring_fallback(missed)
        generic_output = pd.concat(
            [generic_output[generic_output.n_trips.notnull()], ring_fallback]
            )
        # missed = generic_output[generic_output.n_trips.isnull()]
        # if not missed.empty:
        #     all_nzones_fallback = _all_nzones_fallback()
        #     generic_output = pd.concat(
        #         [generic_output[generic_output.n_trips.notnull()],
        #          all_nzones_fallback]
        #         )

    return generic_output


def _process_long_single(sale_id: Tuple,
                         frame: pd.core.frame.DataFrame,
                         takst: str) -> pd.core.frame.DataFrame:

    sub_frame = frame.query("NR in @sale_id")
    sub_frame = sub_frame.query("takstsæt == @takst")

    long = sub_frame.query("betaltezoner > 8")

    long_gen = _long_generic(long)

    long_operator = _long_specific_operator(long)

    return pd.concat([long_gen, long_operator])

def _process_single(
        sales_idxs: Dict,
        frame: pd.core.frame.DataFrame,
        takst: str
        ) -> pd.core.frame.DataFrame:

    short_ids = \
        sales_idxs.get('enkeltbillet', ()) + \
            sales_idxs.get('print-selv-billet', ()) + \
                sales_idxs.get('turistbillet', ())
    short_output = _process_short_single(
        short_ids, frame, takst
        )

    long_ids = \
        sales_idxs.get('enkeltbillet', ()) + \
            sales_idxs.get('lang enkeltbillet', ())

    long_output = _process_long_single(
        long_ids, frame, takst
        )

    single_output = pd.concat([short_output, long_output])

    return single_output

# =============================================================================
# Pendler tickets
# =============================================================================

def _load_kombi_shares() -> pd.core.frame.DataFrame:

    filename = r'sjælland\pendlerkeys2019.csv'

    df = pd.read_csv(filename, index_col=0)
    df.index.name = 'valgtezoner'
    df = df.reset_index()
    df.rename(columns={'S-tog': 'stog'}, inplace=True)

    return df

def _load_kombi_map_shares() -> pd.core.frame.DataFrame:

    filename = r'sjælland\zone_relation_keys2019.csv'
    df = pd.read_csv(filename)
    df = _check_names(df)
    df.loc[:, 'betaltezoner'] = df.loc[:, 'betaltezoner'].fillna(0)
    df.loc[:, 'betaltezoner'] = df.loc[:, 'betaltezoner'].astype(int)

    df = df.drop(['ValidityZones', 'ValidZones'], axis=1)

    df = _stringify_merge_cols(df)

    return df


def _load_nzone_shares():

    filename = r'sælland\kombi_zones_all_trips.csv'

    df = pd.read_csv(filename, index_col=0)
    df.index.name = 'betaltezoner'
    df = df.reset_index()
    df = _check_names(df)
    df = _stringify_merge_cols(df)
    return df

def _chosen_fallback(nullframe):

    nzones = _load_nzone_shares()

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

def _sub_kombi(sub_frame):

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
        chosenfall =  _chosen_fallback(missed)
        chosen_merge = pd.concat(
            [chosen_merge[chosen_merge.n_trips.notnull()], chosenfall]
            )

    return chosen_merge

def _kombi_mappable_pendler(sale_id, frame, takst):

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
            chosenfall =  _chosen_fallback(missed)
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
            recursive_fallback = _sub_kombi(missed)
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
        takst: str
        ) -> pd.core.frame.DataFrame:

    sub_frame = frame.query("NR in @sale_id")
    sub_frame = sub_frame.query("takstsæt == @takst")
    sub_frame = sub_frame.loc[
        (sub_frame.loc[:, 'startzone'].apply(lambda x: str(x)[1:]) == '000') |
        (sub_frame.loc[:, 'slutzone'].apply(lambda x: str(x)[1:]) == '000')
        ]
    sub_frame = _stringify_merge_cols(sub_frame)

    nzone_shares = _load_nzone_shares()

    nzone_output = pd.merge(
        sub_frame, nzone_shares,
        on=['betaltezoner'],
        how='left'
        )

    return nzone_output

def _process_pendler(
        sales_idxs: Dict,
        frame: pd.core.frame.DataFrame,
        takst: str
        ) -> pd.core.frame.DataFrame:

    # TODO make these a user input option

    kombi_pendler_ids = \
        sales_idxs.get('pendlerkort', ())  + \
            sales_idxs.get('pensionistkort', ())  + \
                sales_idxs.get('ungdomskort vu', ())  + \
                    sales_idxs.get('ungdomskort uu', ()) + \
                        sales_idxs.get('flexcard', ())
    kombi_pendler = _kombi_mappable_pendler(
        kombi_pendler_ids, frame, takst
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
        gen_pendler = _nzone_pendler(general_pendler_ids, frame, takst)
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

    filename = r'sjælland\citypass_shares.csv'
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

    sales_data = _load_sales_data(r'mergesales.xlsx')
    sales_numbers = _sales_ref(sales_data)
    # sources = _location_ref(sales_data)

# =============================================================================

    single_output = []
    for takst in ['th', 'ts', 'dsb', 'th']:
        single = _process_single(sales_numbers, sales_data, takst)
        single_output.append(single)
    single_output = pd.concat(single_output)

    pendler_output = []
    for takst in ['th', 'ts', 'dsb', 'th']:

        pendler = _process_pendler(sales_numbers, sales_data, takst)
        pendler_output.append(pendler)

    pendler_output = pd.concat(pendler_output)

    other_output_h = _process_other(sales_numbers, sales_data, 'th')


    found = set(single_output.NR).union(set(pendler_output.NR)).union(
        set(other_output_h.NR))
    missing = sales_data.query("NR not in @found")

    # missing = missing.query("takstsæt == 'th'")

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
