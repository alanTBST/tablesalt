# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:44:17 2020

@author: alkj
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd                      #type: ignore
from pandas.core.frame import DataFrame  #type: ignore


THIS_DIR = Path(os.path.join(os.path.realpath(__file__))).parent

OPERATORS: List[str] = [
    'dsb',
    'first',
    'stog',
    'metro',
    'movia'
    ]


def load_model_results(year: int, model: int) -> DataFrame:

    base = THIS_DIR.parent

    fp = os.path.join(
        base,
        'scripts',
        '__result_cache__',
        f'{year}',
        'output',
        f'takst_sjælland{year}_model_{model}.csv'
        )

    df = pd.read_csv(fp)

    return df


def load_base_factors() -> Dict[str, Dict[str, float]]:

    base_factors = pd.read_csv(
        'zone2_request.csv',
        index_col=0
        ).T.to_dict()


    return base_factors

def weight_trips(base_factors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

    to_weight = ['contains_zone1', 'in_01_or_0103_or_0104'] # A, B

    for k, v in base_factors.items():
        if k in to_weight:
            for op in OPERATORS:
                v[op] = v[op] * v['n_trips']

    return base_factors


def calculate_correction_factors(base_factors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

    A, B = ['contains_zone1', 'in_01_or_0103_or_0104'] # A, B
    C = 'contains_zone2'
    d = {}
    for op in OPERATORS:
        d[op] = base_factors[A][op] - base_factors[B][op]

    total = sum(d.values())
    d = {k: v/total for k, v in d.items()}

    e = {op: base_factors[C][op] / d[op] for op in OPERATORS}
    base_factors['D'] = d
    base_factors['E'] = e

    return base_factors

def calibrate_sale(sale: 'pd.core.frame.sale', base_factors:  Dict[str, Dict[str, float]]) -> Dict[str, float]:

    s = sale._asdict()

    out = {}
    for op in OPERATORS:
        op_revenue = s[op] * s['omsætning']
        try:
            f = op_revenue / s['omsætning']
            g = base_factors['E'][op] * f
            out[op] = g
        except ZeroDivisionError:
            out[op] = s[op]

    total = sum(out.values())
    out = {k: v / total for k, v in out.items()} # calbrate to 100%
    return out


def main(year: int, model: int) -> None:

    basefactors = load_base_factors()
    basefactors = weight_trips(basefactors)
    basefactors = calculate_correction_factors(basefactors)

    results = load_model_results(year, model)

    to_change = results.loc[
        results.loc[:, 'note'].str.contains('INVALID_KOMBI')
        ]

    to_change = to_change.query(
        "note != 'INVALID_KOMBI->kombimatch'"
        )

    sales = to_change.itertuples(index=False, name='sale')

    res = {}
    for sale in sales:
        res[sale.NR] = calibrate_sale(sale, basefactors)

    df = pd.DataFrame.from_dict(res, orient='index')
    df.index.name = 'NR'
    df = df.reset_index()

    df = df[['NR'] + OPERATORS]
    df.to_csv(f'zone2_request_model_{model}.csv')



if __name__ == "__main__":

    main(2019, 3)