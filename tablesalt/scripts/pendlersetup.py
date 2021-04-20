"""
TBST - Trafik, Bygge, og Bolig -styrelsen


Created on Sat Mar 14 18:07:24 2020

@author: Alan Jones
@email: alkj@tbst.dk; alanksjones@gmail.com

# =============================================================================
 HELLO AGAIN!
 -------------
 This script sets up the

 WHAT DOES IT DO?
 -----------------


 Directory tree structure:

 rejsekortstores/
    |
    |------dbs/
    |       |-----kombi_dates_db
    |       |-----kombi_valid_trips
    |       |-----trip_card_db
    |       |-----user_trips_db
    |
    |------hdfstores/
    |       |-----rkfile(0).h5.....rkfile(n).h5
    |
    |------packs/
            |-----rkfile(0)cont.msgpack...rkfile(n)cont.msgpack

 First
 ------

 Second
 -------


  Third
  ------


  BE AWARE!
  ---------


  DEPENDENCIES
  ------
  tablesalt -
  If using conda and the install step doesn't quite work, after the build
  try pip install .  in the repo directory

# =============================================================================
"""
# standard imports
import ast
from datetime import datetime
from functools import partial
# from multiprocessing.pool import ThreadPool
from itertools import groupby
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import DictList

import lmdb
import numpy as np
from pandas import Timestamp

from tablesalt import StoreReader
from tablesalt.common import make_store
from tablesalt.preprocessing import db_paths, find_datastores
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.season import users


def get_pendler_trips(pendler_cards, tripcarddb, userdb):

    user_card_nums = {bytes(x, 'utf-8') for x in pendler_cards}

    trip_card_dict = {}
    with lmdb.open(tripcarddb) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                if v in user_card_nums:
                    trip_card_dict[int(k.decode('utf-8'))] = v

    trip_card_dict = sorted(zip(
        trip_card_dict.values(), trip_card_dict.keys()
        ), key=itemgetter(0))


    card_to_trips = {
        key: bytes(str(tuple(x[1] for x in grp)), 'utf-8')
        for key, grp in groupby(
            trip_card_dict, key=itemgetter(0)
                )
        }

    make_store(card_to_trips, userdb, start_size=5)

    return [x[1] for x in trip_card_dict]


def load_store_dates(store: str, pendler_trip_keys: List[int]) -> Dict[bytes, bytes]:
    """load the time/date data from the given store

    :param store: the path of and hdf5 file
    :type store: str
    :param pendler_trip_keys: a list of tripkeys that are pendler trips
    :type pendler_trip_keys: List[int]
    :return: a dictionary of tripkey -> datestring
    :rtype: Dict[bytes, bytes]
    """

    time_info = StoreReader(store).get_data('time')
    time_info = time_info[np.isin(time_info[:, 0], pendler_trip_keys)]
    date_info = time_info[:, (0, 2, 3, 4, 5)]
    date_info = (tuple(x) for x in date_info)
    date_info = ((x[0], datetime(*x[1:]).date().strftime('%Y-%m-%d'))
                      for x in set(date_info))

    return {
        bytes(str(x[0]), 'utf-8'): bytes(x[1], 'utf-8')
        for x in date_info
        }

def thread_dates(
    lst_of_stores: List[str],
    pendler_trip_keys: List[int],
    dbpath: str
    ) -> None:
    """load all of the stores in parallel, filter pendler trips and write
    to key-value store

    :param lst_of_stores: list of paths to hdf5 files
    :type lst_of_stores: List[str]
    :param pendler_trip_keys: a list of tripkeys that are pendler trips
    :type pendler_trip_keys: List[int]
    :param dbpath: the path of the lmdb key-value store to create
    :type dbpath: str
    """
    func = partial(load_store_dates, pendler_trip_keys=pendler_trip_keys)
    print("Loading travel dates...")
    with Pool(7) as pool:
        results = pool.imap(func, lst_of_stores)
        for res in results:
            make_store(res, dbpath, start_size=5)

def _card_periods(dict_dicts):
    """get val periods from dict of dicts"""
    return set((v['start'].date(), v['end'].date())
                for _, v  in dict_dicts.items())

def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)


def validate_travel_dates(
        userdata: Dict[str, Dict[int, Dict[str, Union[Timestamp, Tuple[int, ...]]]]],
        userdbpath: str,
        kombidatespath: str,
        kombivalidpath: str
        ) -> None:
    """Validate that trips occur in the valid date range for each seasonpass
    write only to ones that do into a key-vaue store


    :param userdata: the dictionary of users and seasonpasses
    :type userdata: Dict[str, Dict[int, Dict[str, Union[Timestamp, Tuple[int, ...]]]]]
    :param userdbpath: the path to the user database key-value store
    :type userdbpath: str
    :param kombidatespath: the path to the key-value store of kombi dates
    :type kombidatespath: str
    :param kombivalidpath: the path to the key value store of valid kombi trips
    :type kombivalidpath: str
    """
    # =========================================================================
    # load the user trips from lmdb
    # =========================================================================
    with lmdb.open(userdbpath) as env:
        usertrips = {}
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                usertrips[k.decode('utf-8')] = \
                    ast.literal_eval(v.decode('utf-8'))
    # =========================================================================
    # validate the user trips using the kombi dates lmdb store
    # =========================================================================
    with lmdb.open(kombidatespath) as env:
        valid_user_dict = {}
        for k, v in usertrips.items():
            try:
                userdates = _card_periods(userdata[k])
            except KeyError:
                continue
            valid_user_trips = []
            with env.begin() as txn:
                for trip in v:
                    trip_date = txn.get(str(trip).encode('utf-8'))
                    if not trip_date:
                        continue
                    trip_date = trip_date.decode('utf-8')
                    trip_date = datetime.strptime(trip_date, '%Y-%m-%d').date()
                    if any(_date_in_window(x, trip_date) for x in userdates):
                        valid_user_trips.append(trip)
            valid_user_dict[k] = tuple(valid_user_trips)

    make_store(valid_user_dict, kombivalidpath, start_size=5)


def main():

    parser = TableArgParser('year', 'products', 'zones')

    args = parser.parse()

    paths = db_paths(find_datastores('H:/'), args['year'])
    stores = paths['store_paths']

    pendler_cards = users._PendlerInput(
        args['year'],
        products_path=args['products'],
        product_zones_path=args['zones']
        )

    print("loading user data")
    userdata = pendler_cards.get_user_data()

    pendler_trip_keys = get_pendler_trips(
        userdata, paths['trip_card_db'], paths['user_trips_db']
        )

    thread_dates(
        stores, pendler_trip_keys, paths['kombi_dates_db']
        )

    print("validating travel dates")
    validate_travel_dates(
        userdata,
        paths['user_trips_db'],
        paths['kombi_dates_db'],
        paths['kombi_valid_trips']
        )
    return

if __name__ == "__main__":
    st = datetime.now()
    main()
    print(datetime.now() - st)
