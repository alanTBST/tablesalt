"""
Contains the RejsekortStore class to interact with all hdf5 datastores
"""

# standard imports
import os
import glob
from datetime import datetime
from multiprocessing import Pool
from typing import Tuple, Set, Dict, Optional, List

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import msgpack

from .storereader import StoreReader


def get_zero_price(price_array: np.ndarray) -> Tuple[int,...]:
    """given an array of price_information, find the trips
    that have a no charge for the trip

    :param price_array: the price array return from an h5 file
    :type price_array: np.ndarray
    :return: a tuple of unique tripkeys of zero price trips
    :rtype: Tuple[int,...]
    """

    df = pd.DataFrame(
        price_array[:, (0, -1)],
        columns=['tripkey', 'price']
        ).set_index('tripkey')

    df = df.groupby(level=0)['price'].transform(max)
    df = df[df == 0]
    return tuple(set(df.index.values))

def load_zero_price(store: str) -> Tuple[int,...]:
    """load and return tripkeys of trips that have zero price

    :param store: the path to an h5 file
    :type store: str
    :return:  a tuple of unique tripkeys of zero price trips
    :rtype: Tuple[int,...]
    """

    price = StoreReader(store).get_data('price')

    return get_zero_price(price)

def find_zero_pay_trips(stores: List[str]) -> Set[int]:
    """load all of the tripkey from the datastores that

    :param stores: a list of paths to h5 files
    :type stores: List[str]
    :return: a set of tripkeys of trips with zero price
    :rtype: Set[int]
    """

    out = set()
    with Pool(os.cpu_count() - 1) as pool:
        results = pool.imap(load_zero_price, stores)
        for res in tqdm(results, 'finding trips inside zones', total=len(stores)):
            out.update(set(res))
    return out


class DataSetError(ValueError):
    """
    ValueError that raises when file in the filepath
    passed to the RejsekortStore does not contain
    the correct datasets
    """

    def __init__(self, missing_sets):
        msg = ("A rejsekort store must contain at least three"
               "datasets: stop_information, time_information and"
               f"passenger information. The store is missing {missing_sets}."
               )
        ValueError.__init__(self, msg)
        self.msg = msg


DEFAULT_FILTERS = {
    'year': None,
    'month': None,
    'day': None,
    'hour': None,
    'minute': None,
    'day_type': None,
    'card_type': None,
    'n_zones': None,
    'exclude_model': None,
    'passenger_types': None
    }


def _find_available_stores(store_dir: str) -> Dict[str, List[str]]:
    """
    Find the available datastores in the directory.

    Args:
        store_dir (str/pathlike): The path of the directory containing
        the rejsekort datastores

    Returns:
        list: list of tuples of the available sub stores

    """
    if not os.path.isdir(os.path.join(store_dir, 'hdfstores')):
        raise OSError("hdfstores directory must be in the datastore path")

    hdf = sorted(glob.glob(
        os.path.join(store_dir, 'hdfstores', '*.h5'))
        )
    mpacks = glob.glob(
        os.path.join(store_dir, 'packs', '*.msgpack')
        )

    mpacks = [x for x in mpacks if 'ZONES' not in x
              and 'CARD_TRIPS' not in x]
    mpacks = sorted(mpacks)

    assert len(mpacks) == len(hdf)

    return {'stores': hdf,
            'packs': mpacks}

class RejsekortStore():

    DEFAULT_DATASTORE_DIR = os.path.join('...', 'datastores')

    def __init__(self, datastore_path: Optional[str] = None, **kwargs) -> None:
        """Class to interact with h5 files of Rejsekort data

        :param datastore_path:the path to the datastore directory, defaults to None
        :type datastore_path: Optional[str], optional
        :raises OSError: if the rejsekort datastores cannot be located
        """

        if datastore_path is not None:
            self._datastore_path = datastore_path
        else:
            try:
                self._year = kwargs.pop('year', None)
                print('did year')
                self._datastore_path = os.path.join(
                    self.DEFAULT_DATASTORE_DIR, f'{self._year}DataStores'
                    )
            except KeyError:
                raise OSError("cannot locate datastores")
        self.sub_stores = self._available_stores()

        self._readers = [StoreReader(x, **kwargs) for x
                         in self.sub_stores['stores']]


    def __repr__(self):
        return f'RejsekortStore at "{self._datastore_path}"'

    def _available_stores(self) ->Dict[str, List[str]]:
        """
        get the lists of substores in the RejsekortStore

        """
        return _find_available_stores(self._datastore_path)

    def _passenger_dicts(self, totals=False, all_types=True):
        """


        Args:
            totals (TYPE, optional): DESCRIPTION. Defaults to False.
            all_types (TYPE, optional): DESCRIPTION. Defaults to True.

        Raises:
            ValueError: DESCRIPTION.

        Returns:
            pas1 (TYPE): DESCRIPTION.
            pas2 (TYPE): DESCRIPTION.
            pas3 (TYPE): DESCRIPTION.

        """
        if totals and all_types:
            raise ValueError("kwargs totals and all_types"
                             "can't both be set to True")
        with h5py.File(self.path) as store:
            pas = store['passenger_information']
            pas1 = {x[0]: x[2] for x in pas}
            pas2 = {x[0]: x[3] for x in pas}
            pas3 = {x[0]: x[4] for x in pas}

        return pas1, pas2, pas3

    def passenger_dict(self) -> Dict[int, int]:
        """
        return total passenger counts as
        a dictionary for the store

        Returns:
            dict: DESCRIPTION.

        """

        with h5py.File(self.path) as store:
            pas = store['passenger_information']
            totals = {x[0]: sum(x[2:5]) for x in pas}
        return totals

    def array_to_classify(self):

        #check keys
        with h5py.File(self.path) as store:
            dsets = store.keys()
            missing_sets = []
            for x in ['stop_information', 'time_information', 'passenger_information']:
                if not x in dsets:
                    missing_sets.append(x)
            if missing_sets:
                raise DataSetError(missing_sets)
            if not self.filters:
                stops = store['stop_information'][:]
                time = store['time_information'][:]
                pas = store['passenger_information'][:]

        stops = stops[np.lexsort((stops[:, 1], stops[:, 0]))]
        time = time[np.lexsort((time[:, 1], time[:, 0]))]
        unix = np.zeros((len(time), 1), dtype=int)
        unix[:, 0] = np.array([int(datetime(*x[2:]).timestamp()) for x in time])

        arr = np.hstack([stops, unix])
        arr = arr[:, [0, 1, -1, 2, 3]]
        arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
        return arr, pas
