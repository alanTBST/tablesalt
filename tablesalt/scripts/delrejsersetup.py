# -*- coding: utf-8 -*-
"""
TBST Trafik, Bygge, og Bolig -styrelsen

Created on Sat Mar 14 18:07:24 2020

@author: Alan Jones
@email: alkj@tbst.dk; alanksjones@gmail.com

# =============================================================================
 HELLO THERE!
 -------------

 This script is the first step in any analysis of delrejser data
 at TBST using Python. It creates the datastores that are needed for both the
 OD analysis and the Revenue distribution for DOT for Takstsjælland

 WHAT DOES IT DO?
 -----------------

 Given a path to a directory of compressed zip files of rejsekort delrejser
 data and a path for an output directory where the resulting datastores
 should be placed, it will read those zips without having to extract them
 first and splits the large dataset into various forms and files.

 Most significantly it will split the giant data set into hundreds of smaller
 files.

 Directory tree structure:

 GIVEN_OUTPUT_DIRECTORY/
          |
          |
          |---rejsekortstores/
                    |
                    |------dbs/
                    |       |-----trip_card_db (key-value_store)
                    |
                    |------hdfstores/
                    |         |-----rkfile(0).h5
                    |         |----- ...
                    |         |-----rkfile(n).h5
                    |------packs/
                    |        |-----rkfile(0)cont.msgpack
                    |        |----- ...
                    |        |-----rkfile(n)cont.msgpack
 First
 ------
 A key-value store with tripkeys as keys and card numbers as values in a
 memory mapped lmdb database for super fast lookups.
 This database contains all the tripkeys in the dataset.
 For 2019 delrejser data, this uses approximately 5gb. The size is flexible
 up until 30gb without any user input, although that upper limit can be
 changed.

 Second
 -------
 The flat data set is split up into more coherent subsets:
     - stop_information
     - time_information
     - price_information
     - passenger_information
     - contractor_information

  The first four dataset are places in hdf5 files in the directory structure
  shown above. Each of these have been normalised and contain only integers
  hdf5 files store and load matrices efficiently

  Third
  ------
  The contractor information currently is put in msgpack files. These
  files are similar to json but are smaller and load quickly.
  The contractor_information is semi-normalised but contains the from and
  to stop variables as strings. (This may change in the future to a
  document database in the future)

  Using this structure, analysis/computation can be easily parallelised
  and reduces the reliance on RDBMS (SQL Server) database connections -
  which can be slow

  BE AWARE!
  ---------
  This script spawns four process, so ensure that you have four cores
  available. If using a laptop, for instance, you won't be able to get
  much else done. It shouldn't last much more than 7 hrs though :)
  Multi-core server architecture advised.

  DEPENDENCIES
  ------
  tablesalt - tbst python analysis package

  numpy, pandas, python-lmdb, h5py, msgpack
  Each of these can be installed with pip or conda


# =============================================================================
"""
# standard imports
import os
import zipfile
from threading import Thread
from datetime import datetime
import time
import json
import site

from itertools import groupby
from multiprocessing import Process, Queue
from operator import itemgetter
from typing import (
    ByteString,
    Iterator,
    List,
    Dict,
    Tuple,
    Optional,
    AnyStr,
    Union,
    Any
    )

# third party imports
import numpy as np       #type: ignore
import pandas as pd      #type: ignore
import h5py              #type: ignore
import msgpack           #type: ignore
from tqdm import tqdm    #type: ignore

# this package imports
from tablesalt.common.io import mappers
from tablesalt.common import make_store
from tablesalt.running import WindowsInhibitor
from tablesalt.preprocessing.parsing import TableArgParser
from tablesalt.preprocessing.tools import (
    setup_directories,
    get_zips,
    check_all_file_headers,
    col_index_dict,
    get_columns,
    sumblocks
    )


# =============================================================================
# lmdb creation
# =============================================================================
def card_trip_generator(
        ziplist, col_indices, chunksize, skiprows=0
        ) -> Iterator[Dict[bytes, bytes]]:
    """
    Parameters
    ----------
    ziplist : list
        A list of the zipfiles and contents
    col_indices : dict
        a dictionary of column names and corresponding indices.
    chunksize : int, optional
        the chunksize to use.
    skiprows : int, optional
        the number of rows to skip in the file. The default is 0.

    Yields
    -------
    dict :
        the generator for the lmbd database.

    """
    wanted_columns = [col_indices['kortnr'], col_indices['turngl']]

    for zfile, content in ziplist:
        df_gen = pd.read_csv(
            zipfile.ZipFile(zfile).open(content),
            encoding='iso-8859-1', usecols=wanted_columns,
            chunksize=chunksize, skiprows=skiprows,
            error_bad_lines=False, low_memory=False
            )

        for chunk in df_gen:
            chunk_dict = dict(zip(chunk.iloc[:, 0], chunk.iloc[:, 1]))
            byte_dict = {bytes(str(k), 'utf-8'): bytes(str(v), 'utf-8')
                         for k, v in chunk_dict.items()}
            yield byte_dict

def make_tripcard_kvstore(zips, col_indices, location, chunksize):
    """create the lmdb key-value store for trips and cardnums"""
    dbpath = os.path.join(location, 'trip_card_db')

    for i, bytedict in enumerate(
            card_trip_generator(zips, col_indices, chunksize)
        ):
        make_store(bytedict, dbpath, start_size=12)

# =============================================================================
# hdf5
# =============================================================================
def delrejser_generator(ziplist, inp_cols, col_indices,
                        chunksize, skiprows=0):
    """
    Yield a np.ndarray (object) generator of the delrejser dataset

    Parameters
    ----------
    zfile : string or path like
        the path to a zipfile of delrejser data.
    content : string
        the name of a file in the zipfile.
    col_indices : dict
        a dictionary of column names and corresponding indices.
    col_dtypes : dict
        a dictionary of datatypes for the colums.
    chunksize : int, optional
        the chunksize to use. The default is 1000000.
    skiprows : int, optional
        the number of rows to skip in the file. The default is 0.

    Returns
    -------
    gen :  np.ndarray generator
        the generator for the delrejser file.

    """
    wanted_columns = [col_indices[x] for x in inp_cols]
    n_files = len(ziplist)
    content_count = 1
    for zfile, content in ziplist:
        nrows = sumblocks(zfile, content)
        # nrows = 301_000_000
        total_chunks =  nrows // chunksize

        df_gen = pd.read_csv(
            zipfile.ZipFile(zfile).open(content),
            encoding='iso-8859-1', usecols=wanted_columns,
            chunksize=chunksize, skiprows=skiprows,
            error_bad_lines=False, low_memory=False
            )
        for df in tqdm(
                df_gen,
                total=total_chunks,
                postfix=f'file {content_count} of {n_files}'
                ):
            try:
                df['tidsrabat'] = df['tidsrabat'].astype(str)
                df['RejsePris'] = df['RejsePris'].astype(str)
                df['tidsrabat'] = df['tidsrabat'].str.replace(',', '')
                df['tidsrabat'] = df['tidsrabat'].str.replace('.', '')
                df['RejsePris'] = df['RejsePris'].str.replace(',', '')
                df['RejsePris'] = df['RejsePris'].str.replace('.', '')
            except KeyError:
                pass
            df = df.fillna(0)
            yield df.values
        content_count += 1


def _resolve_column_orders(input_columns, col_indices):

    wanted_columns = [(x, col_indices[x]) for x in input_columns]
    wanted_columns = sorted(wanted_columns, key=lambda x: x[1])
    return [x[0] for x in wanted_columns]


def format_time(x):
    try:
        time = datetime.fromisoformat(x)
    except AttributeError:
        time = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S")
    return (time.year, time.month, time.day,
            time.hour, time.minute, time.second,
            time.weekday())


def trip_application(array, column_order):
    """
    create two sub arrays in a tuple using trip keys and application
    sequence.
    the first is the tripkey application and time (split to integers)
    and the second is the tripkey application and stopPointNr
    """

    turngl = column_order.index('turngl')
    app = column_order.index('applicationtransactionsequencenu')
    stop_num = column_order.index('stoppointnr')
    model_id = column_order.index('model')
    msgreport = column_order.index('msgreportdate')

    try:
        reversed_model_dict = {v:k for k, v in mappers['model_dict'].items()}
        stop_arr = array[:, (turngl, app, stop_num, model_id)]
        stop_arr[:, 3] = np.vectorize(reversed_model_dict.get)(stop_arr[:, 3])
        time_arr = array[:, (turngl, app)]
        split_time = np.vstack(np.vectorize(format_time)(array[:, msgreport])).T
        time_arr = np.hstack([time_arr, split_time])

        return (np.array(time_arr, dtype=np.int64),
                np.array(stop_arr, dtype=np.int64))
    except Exception as e:
        pass

    try:
        trip_app = np.array(array[:][:, (turngl, app)], dtype=np.int64)
    except ValueError:
        trip_app = np.array(array[:][:, (turngl, app)], dtype=np.float64)
        trip_app = np.array(trip_app, dtype=np.int64)

    stop_ids = np.array(array[:][:, stop_num], dtype=np.int32).T
    model = array[:, model_id]

    reversed_model_dict = {v:k for k, v in mappers['model_dict'].items()}
    try:
        model = np.vectorize(reversed_model_dict.get)(model)
    except TypeError:
        model = np.array([reversed_model_dict.get(x, 0) for x in model])

    trip_app_stop = np.zeros(shape=(len(trip_app), 4),
                             dtype=np.int64)

    time_format = "%Y-%m-%dT%H:%M:%S"
    times = (datetime.strptime(x, time_format) for
             x in array[:, msgreport])

    times_arr = np.array([[x.year, x.month, x.day,
                           x.hour, x.minute, x.second,
                           x.weekday()] for x in times])

    trip_app_stop[:, 0] = trip_app[:, 0]
    trip_app_stop[:, 1] = trip_app[:, 1]
    trip_app_stop[:, 2] = stop_ids
    trip_app_stop[:, 3] = model
    msgreportdate = np.hstack([trip_app, times_arr])

    return np.array(msgreportdate, dtype=np.int64), trip_app_stop

def passenger_information(array: Any, column_order):
    """
    return sub array of tripkey and passenger counts and types
    """
    turngl = column_order.index('turngl')

    pas1 = column_order.index('passagerantal1')
    pas2 = column_order.index('passagerantal2')
    pas3 = column_order.index('passagerantal3')
    ptype1 = column_order.index('passagertype1')
    ptype2 = column_order.index('passagertype2')
    ptype3 = column_order.index('passagertype3')
    ktype = column_order.index('korttype')

    key_pas = array[:, [turngl, pas1, pas2,
                        pas3, ptype1, ptype2,
                        ptype3, ktype]]
    key_pas[key_pas == ''] = '0'
    key_pas[:, 7] = np.vectorize(mappers['card_id'].get)(key_pas[:, 7])
    key_pas = np.array(key_pas, dtype=np.int64)
    key_pas = np.unique(key_pas, axis=0)

    key_pas = key_pas[(key_pas[:, 1] != 0)
                      | (key_pas[:, 2] != 0)
                      | (key_pas[:, 3] != 0)]

    key_pas = key_pas[:, [0, 7, 1, 2, 3, 4, 5, 6]]

    return key_pas


def price_information(array, column_order):
    """
    return sub array of tripkey, rebates, zones travelled and trip price
    """
    turngl = column_order.index('turngl')
    rejsepris = column_order.index('rejsepris')
    # tidsrabat = column_order.index('tidsrabat')
    zonerrejst = column_order.index('zonerrejst')

    tprz = array[:, (turngl,
                      rejsepris,
                      # tidsrabat,
                      zonerrejst)]

    tprz = tprz[tprz[:, 1] != 'nan']
    tprz = tprz[tprz[:, 1] != '0']
    tprz = tprz[tprz[:, 1] != 0]
    tpz = np.array(tprz[:, (0, 1, 2)], dtype=np.int64)

    # price = np.zeros(shape=(len(tprz), 1))

    # price_list = (x.replace('.', '').strip("'")
    #               for x in tprz[:, 1])
    # price_list = (x.replace(',', '.') for x in price_list)
    # price_list = (x.replace('nan', '0') for x in price_list)
    # price_list = (x.replace('NaN', '0') for x in price_list)
    # price_list = [int(float(x) * 100)
    #               for x in price_list]
    # price[:, 0] = np.array(price_list, dtype=np.int64)
    # price_info = np.hstack((trz, price))
    # np.array(price_info, dtype=np.int64)
    return tpz


def check_collection_complete(arr_col, key):
    """cheack the common.io collection mapper is the same"""
    test_vals = mappers[key].keys()
    unseen = set()
    for x in arr_col:
        if x not in test_vals:
            unseen.add(x)
    if not unseen:
        return None
    if any(isinstance(x, int) for x in unseen):
        return None
    return unseen

def update_collection(unseen_ids, key):
    """change the io mappers if needed"""
    # TODO put this updating activity in a class
    current_max_key_id = max(mappers[key].values())

    package_loc = site.getsitepackages()
    collection_loc = os.path.join(
        package_loc[1], 'tablesalt', 'common',
        'io', 'rejsekortcollections.json'
        )
    try:
        with open(collection_loc, 'r', encoding='iso-8859-1') as f:
            old_collection = json.loads(f.read().encode('iso-8859-1').decode())

        for i, x in enumerate(unseen_ids):
            mappers[key][x] = current_max_key_id + 1 + i
            old_collection[key][x] = current_max_key_id + 1 + i

        with open(collection_loc, 'w') as fp:
            json.dump(old_collection, fp)
    except Exception as e:
        print(str(e))
        print("\n")
        print("skipping mappers update")


def contractor_information(array, column_order):
    """
    return sub array of tripkey, AppSeq, the operators and contractors
    """

    turngl = column_order.index('turngl')
    app = column_order.index('applicationtransactionsequencenu')
    nyd = column_order.index('nyudfører')
    contid = column_order.index('contractorid')
    route_id = column_order.index('ruteid')
    fra = column_order.index('fradelrejse')
    til = column_order.index('tildelrejse')

    cont_arr = array[:, (turngl, app, nyd, contid,
                         route_id, fra, til)]

    unseen = check_collection_complete(cont_arr[:, 2], 'operator_id')

    if not unseen:
        cont_arr[:, 2] = [mappers['operator_id'].get(x, 0) for x in cont_arr[:, 2]]
    else:
        update_collection(unseen, 'operator_id')
        cont_arr[:, 2] = [mappers['operator_id'].get(x, 0) for x in cont_arr[:, 2]]
    unseen = check_collection_complete(
        cont_arr[:, 3], 'contractor_id'
        )
    if not unseen:
        cont_arr[:, 3] = [mappers['contractor_id'].get(x, 0) for x in cont_arr[:, 3]]
    else:
        print('updating collection')
        update_collection(unseen, 'contractor_id')
        cont_arr[:, 3] = [mappers['contractor_id'].get(x, 0) for x in cont_arr[:, 3]]

    cont_arr = cont_arr[np.lexsort((cont_arr[:, 1], cont_arr[:, 0]))]

    cont_dict = {
        int(key): tuple(
            (int(x[1]), int(x[2]), int(x[3]), x[4], x[5], x[6]) for x in group
            ) for key, group in groupby(cont_arr, key=itemgetter(0))
        }

    return cont_dict



def store_processing(dset, ziplist, col_indices, chunksize):
    """
    Parameters
    ----------
    dset : TYPE
        DESCRIPTION.
    ziplist : TYPE
        DESCRIPTION.
    col_indices : TYPE
        DESCRIPTION.

    Yields
    ------
    i : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    dset : TYPE
        DESCRIPTION.

    """
    col_dict = {
        'stoptime': [
            'turngl',
            'applicationtransactionsequencenu',
            'stoppointnr', 'model',
            'msgreportdate'
            ],
        'passengers': [
            'turngl', 'passagerantal1',
            'passagerantal2', 'passagerantal3',
            'passagertype1', 'passagertype2',
            'passagertype3', 'korttype'
            ],
        'price': [
            'turngl', 'rejsepris',
            'tidsrabat', 'zonerrejst'
            ],
        'contractor': [
            'turngl',
            'applicationtransactionsequencenu',
            'nyudfører', 'contractorid',
            'ruteid', 'fradelrejse',
            'tildelrejse'
            ]
        }
    func_dict = {
        'stoptime': trip_application,
        'passengers': passenger_information,
        'price' : price_information,
        'contractor': contractor_information
        }

    column_ord = _resolve_column_orders(col_dict[dset], col_indices)

    arr_gen = delrejser_generator(
        ziplist, col_dict[dset], col_indices, chunksize
        )

    for i, arr in enumerate(arr_gen):
        yield i, func_dict[dset](arr, column_ord), dset


def stoptime_producer(ziplist, col_indices, q, chunksize) -> None:

    for x in store_processing('stoptime', ziplist, col_indices, chunksize):
        q.put(x)
        time.sleep(0.1)


def price_producer(ziplist, col_indices, q, chunksize) -> None:

    for x in store_processing('price', ziplist, col_indices, chunksize):
        q.put(x)
        time.sleep(0.1)


def passengers_producer(ziplist, col_indices, q, chunksize) -> None:

    for x in store_processing('passengers', ziplist, col_indices, chunksize):
        q.put(x)
        time.sleep(0.1)


def contractor_producer(ziplist, col_indices, q, chunksize) -> None:
    """
    Parameters
    ----------
    ziplist : TYPE
        DESCRIPTION.
    col_indices : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for x in store_processing(
            'contractor', ziplist, col_indices, chunksize):
        q.put(x)
        time.sleep(0.1)


def consumer_process(q, location) -> None:

    while True:
        if not q.empty():
            res = q.get()
            if res is None:
                break
        else:
            time.sleep(0.1)
            continue


        datasets = []
        out_data = res[1]
        name = res[2]

        store_path = os.path.join(
            location, f'rkfile{res[0]}.h5'
            )
        if name == 'price':
            dsetname = 'price_information'
            datasets.append((out_data, dsetname))
        elif name == 'stoptime':
            datasets.append((out_data[0], 'time_information'))
            datasets.append((out_data[1], 'stop_information'))
        elif name == 'passengers':
            datasets.append((out_data, 'passenger_information'))
        # print(res[0], name)
        with h5py.File(store_path, 'a') as store:
            dt = np.uint64
            for dset in datasets:
                store.create_dataset(
                    data=dset[0], name=dset[1],
                    dtype=dt, compression='gzip',
                    compression_opts=4
                    )


def contractor_consumer(q, location) -> None:
    """
    """
    while True:
        if not q.empty():
            res = q.get()
        else:
            time.sleep(0.1)
            continue
        if res is None:
            break

        out_data = res[1]

        store_path = os.path.join(
            location, f'rkfile{res[0]}cont.msgpack'
            )
        with open(store_path, 'wb') as outfile:
            msgpack.pack(out_data, outfile)

# =============================================================================
#  Start process for each dataset and put in a queue for each chunk
# =============================================================================
def main() -> None:
    """entry"""

    parser = TableArgParser('year', 'chunksize', 'input_dir', 'output_dir')
    args = parser.parse()

    year = args['year']
    output_dir = args['output_dir']
    input_dir = args['input_dir']
    chunk_size = args['chunksize']

    paths = setup_directories(year, output_dir)

    tc_store_path = paths['dbs']
    hdf_path = paths['hdfstores']
    cont_path = paths['packs']

    zips = get_zips(input_dir)
    check_all_file_headers(zips)
    file_columns = get_columns(*zips[0])
    col_indices, a = col_index_dict(file_columns)

    queue = Queue()
    contractor_queue = Queue()

    t1 = Thread(target=make_tripcard_kvstore,
                args=(zips, col_indices, tc_store_path, chunk_size))
    print("procesessing stores")
    p1 = Process(target=stoptime_producer,
                  args=(zips, col_indices, queue, chunk_size))
    p2 = Process(target=price_producer,
                  args=(zips, col_indices, queue, chunk_size))
    p3 = Process(target=passengers_producer,
                  args=(zips, col_indices, queue, chunk_size))
    p4 = Process(target=contractor_producer,
                  args=(zips, col_indices, contractor_queue, chunk_size))

    t2 = Thread(target=consumer_process,
                args=(queue, hdf_path))

    t3 = Thread(target=contractor_consumer,
                args=(contractor_queue, cont_path))

    t1.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    t2.start()
    t3.start()

    t1.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    queue.put(None)
    contractor_queue.put(None)
    t2.join()
    t3.join()

if __name__ == "__main__":


    dt = datetime.now()

    if os.name == 'nt':
        INHIBITOR = WindowsInhibitor()
        INHIBITOR.inhibit()
        main()
        INHIBITOR.uninhibit()
    else:
        main()

    print(datetime.now() - dt)
