# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 22:13:10 2020

@author: alkj
"""

import os
import pickle
from tablesalt.common.connections import make_connection

def _get_rabatkeys(rabattrin, year):

    fp = os.path.join(
        '__result_cache__', 
        f'rabat{rabattrin}trips.pickle'
        )
    try:
        with open(fp, 'rb') as f:
            rabatkeys = pickle.load(f)
    except FileNotFoundError:
        rabatkeys = helrejser_rabattrin(rabattrin, year)

    return rabatkeys

def helrejser_rabattrin(rabattrin, year):
    """
    return a set of the tripkeys from the helrejser data in
    the datawarehouse

    parameter
    ----------
    rabattrin:
        the value of the rabattrin
        int or list of ints
        default is None, returns all

    """

    query = (
        "SELECT Turngl FROM "
        "[dbDwhExtract].[Rejsedata].[EXTRACT_FULL_Helrejser_DG7] "
        f"where [År] = {year} and [Manglende-check-ud] = 'Nej' and "
        f"Produktfamilie = '5' and [Rabattrin] = {rabattrin}"
        )
    # ops = make_options(
    #     prefer_unicode=True,
    #     use_async_io=True
    #     ) # , use_async_io=True
    with make_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        try:
            gen = cursor.fetchnumpybatches()
            try:
                out = set().union(*[set(batch['Turngl']) for batch in gen])
            except KeyError:
                try:
                    out = set().union(*[set(batch['turngl']) for batch in gen])
                except KeyError:
                    raise KeyError("can't find turngl")
        except AttributeError:
            gen = cursor.fetchall()

        out = set(chain(*gen))

    return {int(x) for x in out}
