"""
TBST - Trafik, Bygge, og Bolig -styrelsen

Created on Sat Mar 14 18:07:24 2020

@author: Alan Jones
@email: alkj@tbst.dk; alanksjones@gmail.com

**HELLO AGAIN!**

This script is the second step in revenue analysis at TBST

**WHAT DOES IT DO?**

    Given a paths to pendler product data this script creates three more lmdb
    key-value stores pertienent to pendler users. The delrejsersetup script, run before
    this, created the trip_card_db


Input Data
==========

PeriodeProdukt
--------------

+----------------------------------+--------------+----------------------+---------------------------------+
|          EncryptedCardEngravedID | SeasonPassID | SeasonPassTemplateID | SeasonPassName                  |
+==================================+==============+======================+=================================+
| 42454135373736374541303030343937 |      2120462 |              8192163 | DOT Pendler Kombi m. Metro      |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2097861 |              8192130 | Udgået - Pendler Kombi Sjælland |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2105617 |              8192130 | Udgået - Pendler Kombi Sjælland |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2113630 |              8192130 | Udgået - Pendler Kombi Sjælland |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2125274 |              8192163 | DOT Pendler Kombi m. Metro      |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2133790 |              8192163 | DOT Pendler Kombi m. Metro      |
+----------------------------------+--------------+----------------------+---------------------------------+
| 32433241303531383137334546443541 |      2141567 |              8192163 | DOT Pendler Kombi m. Metro      |
+----------------------------------+--------------+----------------------+---------------------------------+

...

+----------+----------------+------------------+---------------------+
| Fareset  | PsedoFareset   | SeasonPassType   | PassengerGroupType1 |
+==========+================+==================+=====================+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+
| Sjælland | Hovedstaden    | NumberOfZones    |                   1 |
+----------+----------------+------------------+---------------------+

...

+------------------------------------------------------+------------------+----------------+
| SeasonPassStatus                                     | ValidityStartDT  | ValidityEndDT  |
+======================================================+==================+================+
| Slettet, periodekort er allerede fjernet fra kortet. | 03/09/2019       | 16/09/2019     |
+------------------------------------------------------+------------------+----------------+
| Slettet, periodekort er allerede fjernet fra kortet. | 05/06/2019       | 04/07/2019     |
+------------------------------------------------------+------------------+----------------+
| Slettet, periodekort er allerede fjernet fra kortet. | 05/07/2019       | 03/08/2019     |
+------------------------------------------------------+------------------+----------------+
| Slettet, periodekort er allerede fjernet fra kortet. | 05/08/2019       | 03/09/2019     |
+------------------------------------------------------+------------------+----------------+
| Stoppet, periodekort kan fjernes fra kortet          | 12/09/2019       | 11/10/2019     |
+------------------------------------------------------+------------------+----------------+
| Slettet, periodekort er allerede fjernet fra kortet. | 22/10/2019       | 20/11/2019     |
+------------------------------------------------------+------------------+----------------+
| Stoppet, periodekort kan fjernes fra kortet          | 21/11/2019       | 20/12/2019     |
+------------------------------------------------------+------------------+----------------+

...

+-------------+--------------+------------+-----------------+--------------+----------------+
|   ValidDays |   FromZoneNr |   ToZoneNr | SeasonPassZones | PassagerType | TpurseRequired |
+=============+==============+============+=================+==============+================+
|          14 |         1002 |       1052 |               5 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+
|          30 |         1001 |       1085 |               8 | Voksen       | Yes            |
+-------------+--------------+------------+-----------------+--------------+----------------+

...

+---------------------+----------+---------------------------+--------------+
| SeasonPassCategory  | Pris     | RefundType                | productdate  |
+=====================+==========+===========================+==============+
| Standard            | 623,26   | TILBAGEBETALING REJSEKORT | 03/09/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 05/06/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 05/07/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 05/08/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 12/09/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 22/10/2019   |
+---------------------+----------+---------------------------+--------------+
| Standard            | 1.329,90 | nan                       | 21/11/2019   |
+---------------------+----------+---------------------------+--------------+


Zoner
-----

+----------------------------------+----------------+----------+
|          EncryptedCardEngravedID |   SeasonPassID |   ZoneNr |
+==================================+================+==========+
| 42454135373736374541303030343937 |        2120462 |     1001 |
+----------------------------------+----------------+----------+
| 42454135373736374541303030343937 |        2120462 |     1002 |
+----------------------------------+----------------+----------+
| 42454135373736374541303030343937 |        2120462 |     1031 |
+----------------------------------+----------------+----------+
| 42454135373736374541303030343937 |        2120462 |     1041 |
+----------------------------------+----------------+----------+
| 42454135373736374541303030343937 |        2120462 |     1052 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1001 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1002 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1031 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1042 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1053 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1063 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1074 |
+----------------------------------+----------------+----------+
| 32433241303531383137334546443541 |        2097861 |     1085 |
+----------------------------------+----------------+----------+

Resultant directory tree structure
==================================

| rejsekortstores/
|
|    |------dbs/
|           |-----trip_card_db
|           |-----kombi_dates_db
|           |-----kombi_valid_tripshat
|           |-----user_trips_db
|
|    |------hdfstores/
|           |-----rkfile(0).h5.....rkfile(n).h5
|
|    |------packs/
|           |-----rkfile(0)cont.msgpack...rkfile(n)cont.msgpack



"""
import ast
from datetime import datetime
from functools import partial
from itertools import groupby
from multiprocessing import Pool
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Set, Union, Tuple

import lmdb
import numpy as np
from pandas import Timestamp

from tablesalt import StoreReader
from tablesalt.common import make_store
from tablesalt.preprocessing import db_paths, find_datastores
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.season import users

# may refactor to accept just a sequence of cardnums
def get_pendler_trips(
    userdata: Dict[str, Dict[int, Dict[str, Union[Timestamp, Tuple[int, ...]]]]],
    tripcarddb: str, userdb: str
    ) -> List[str]:
    """create the user_trips_db from the season pass data and the trip_card kv store
        cardnum -> (tripkey1, tripkey2, tripkey3,....)
        also returns a list of pendler tripkeys

    :param userdata: the data returned from users._PendlerInput.get_user_data()
    :type userdata: Dict[str, Dict[int, Dict[str, Union[Timestamp, Tuple[int, ...]]]]]
    :param tripcarddb: the path to the tripcarddb
    :type tripcarddb: str
    :param userdb: the path the output user_trips_db
    :type userdb: str
    :return: a list of all pendler tripkeys
    :rtype: List[int]
    """

    user_card_nums = {bytes(x, 'utf-8') for x in userdata}

    trip_card_dict = {}
    with lmdb.open(tripcarddb) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                if v in user_card_nums:
                    trip_card_dict[int(k.decode('utf-8'))] = v

    trip_card_list = sorted(zip(
        trip_card_dict.values(), trip_card_dict.keys()
        ), key=itemgetter(0))


    card_to_trips = {
        key: bytes(str(tuple(x[1] for x in grp)), 'utf-8')
        for key, grp in groupby(
            trip_card_list, key=itemgetter(0)
                )
        }

    make_store(card_to_trips, userdb, start_size=5)

    return [trip for card, trip in trip_card_list]


def load_store_dates(store: str, pendler_trip_keys: List[int]) -> Dict[bytes, bytes]:
    """load the time/date data from the given store and get only pendler user
    tripkeys

    :param store: the path of an hdf5 file
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
    """main script function
    """

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
        userdata, paths['trip_card_db'],
        paths['user_trips_db']
        )

    thread_dates(
        stores,
        pendler_trip_keys,
        paths['kombi_dates_db']
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
