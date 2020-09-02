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
import sys
import os
import glob
from datetime import datetime
from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
from itertools import groupby
from functools import partial
from operator import itemgetter
from pathlib import Path
from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
    )


#thrid party imports
import numpy as np
import lmdb

#package imports
from tablesalt import StoreReader
from tablesalt.season import users
from tablesalt.common import make_store
from tablesalt.preprocessing import find_datastores, db_paths

def parse_args():
    """parse the cl arguments"""
    DESC = ("Setup all the key-value stores needed \n"
            "for the pendler card revenue distribution \n"
            "for takstsjælland.")

    parser = ArgumentParser(
        description=DESC,
        formatter_class=RawTextHelpFormatter
        )
    parser.add_argument(
        '-y', '--year',
        help='year to unpack',
        type=int,
        required=True
        )
    parser.add_argument(
        '-z', '--zones',
        help='path to input zones csv',
        type=Path,
        required=True
        )
    parser.add_argument(
        '-p', '--products',
        help='path to input pendler products csv',
        type=Path,
        required=True
        )

    args = parser.parse_args()

    return vars(args)

def _hdfstores(store_loc, year):

    return glob.glob(
        os.path.join(
            store_loc, 'rejsekortstores',
            f'{year}DataStores', 'hdfstores', '*.h5'
            )
        )


def get_pendler_trips(pendler_cards, tripcarddb, userdb):
    """
    Filter out non-pendler kombi trips

    Parameters
    ----------
    card_trip_fp : str/path
        path to the CARD_.
    pendler_cards : dict
        the pendler user data returned from
        users._PendlerInput().get_user_data()

    Returns
    -------
    list
        list of tuples.

    """

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


def load_store_dates(store, pendler_trip_keys):
    """
    load the time information from the hdfstores

    Parameters
    ----------
    store : str/path
        path to an hdfstore.
    pendler_trip_keys : list
        a list of tripkey integers.

    Returns
    -------
    tuple
        tuple of two-tuples of (tripkey, date string).

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

def thread_dates(lst_of_stores, pendler_keys, dbpath):
    """
    Load the pendler

    Parameters
    ----------
    card_trip_fp : str
        DESCRIPTION.
    pendler_cards : dict
        DESCRIPTION.
    stores : TYPE
        DESCRIPTION.
    n_threads : TYPE, optional
        DESCRIPTION. The default is 4.

    Returns
    -------
    list
        list of two-tules (tripkey, datetime.date).

    """
    func = partial(load_store_dates, pendler_trip_keys=pendler_keys)
    print("Loading travel dates...")
    with Pool(3) as pool:
        results = pool.imap(func, lst_of_stores)
        for res in results:
            make_store(res, dbpath, start_size=5)
            env = lmdb.open(dbpath)
            print(env.stat()['entries'])
            env.close()

def _card_periods(dict_dicts):
    """get val periods from dict of dicts"""
    return set((v['start'].date(), v['end'].date())
                for _, v  in dict_dicts.items())

def _date_in_window(test_period, test_date):
    """test that a date is in a validity period"""
    return min(test_period) <= test_date <= max(test_period)


def validate_travel_dates(
        userdata, userdbpath,
        kombidatespath, kombivalidpath
        ):
    """
    Validate that a trip occurs in a valid kombi time period

    Parameters
    ----------
    travel_dates : TYPE
        DESCRIPTION.
    userdata : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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

    args = parse_args()

    year = args['year']
    zone_path = args['zones']
    product_path = args['products']

    store_dir = find_datastores(r'H://')
    paths = db_paths(store_loc, year)
    store_files = _hdfstores(store_dir, year)

    pendler_cards = users._PendlerInput(
        year, products_path=product_path,
        product_zones_path=zone_path
        )

    print("loading user data")
    userdata = pendler_cards.get_user_data()

    pendler_trip_keys = get_pendler_trips(
        userdata, paths['trip_card_db'], paths['user_trips_db']
        )

    thread_dates(
        store_files, pendler_trip_keys, paths['kombi_dates_db']
        )

    print("validating travel dates")
    validate_travel_dates(
        userdata, paths['user_trips_db'],
        db_dirs['kombi_dates_db'], paths['kombi_valid_trips']
        )
    return

if __name__ == "__main__":
    st = datetime.now()
    main()
    print(datetime.now() - st)
