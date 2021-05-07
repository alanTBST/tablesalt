# -*- coding: utf-8 -*-
"""
Module containg StoreReader class to read
datasets from the Rejsekort datasets that
are created from the csv zipfile delivered
by rejsedata
"""

import os
import pathlib
from itertools import chain, groupby
from operator import itemgetter
from typing import Iterable, Dict, Set, Union, List, Tuple, Any


import h5py #type: ignore
import msgpack #type: ignore
import numpy as np #type: ignore

from tablesalt.common.io.rejsekortcollections import (_load_collection,
                                                      proc_collection)


mappers = _load_collection()
mappers = proc_collection(mappers)

DSET_NAME_MAP = {
    'stops': 'stop_information',
    'stop_information': 'stop_information',
    'stop': 'stop_information',
    'price': 'price_information',
    'prices': 'price_information',
    'price_information': 'price_information',
    'times': 'time_information',
    'time': 'time_information',
    'time_information': 'time_information',
    'pas': 'passenger_information',
    'passengers': 'passenger_information',
    'passenger': 'passenger_information',
    'passenger_information': 'passenger_information',
    'cont': 'contractor_information',
    'contr': 'contractor_information',
    'contractors': 'contractor_information',
    'contractor': 'contractor_information',
    'operator': 'contractor_information',
    'operators':  'contractor_information',
    'contractor_information': 'contractor_information'
    }

# TODO...put in config
PAS_TYPES = {
    'voksen': [1, 7, 8, 11],
    'barn': [2, 9, 10, 12],
    'cykel': [3, 13, 16, 20],
    'handicap': [18],
    'hund': [6, 14, 17],
    'pensionist': [4, 19],
    'ung': [5],
    'main': [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 18, 19],
    'all': list(range(21))
    }

PAS_TYPES['youth'] = PAS_TYPES['ung']
PAS_TYPES['adult'] = PAS_TYPES['voksen']
PAS_TYPES['bicycle'] = PAS_TYPES['cykel']
PAS_TYPES['dog'] = PAS_TYPES['hund']
PAS_TYPES['child'] = PAS_TYPES['barn']
PAS_TYPES['pensioner'] = PAS_TYPES['pensionist']
PAS_TYPES['bike'] = PAS_TYPES['cykel']

OP_IDS = mappers['operator_id'].copy()
OP_IDS['D'] = OP_IDS['DSB']
OP_IDS['dsb'] = OP_IDS['DSB']
OP_IDS['A'] = OP_IDS['Arriva']
OP_IDS['S'] = OP_IDS['Stog']
OP_IDS['Stog'] = OP_IDS['Stog']
OP_IDS['Movia'] = [OP_IDS['Movia_H'],
                   OP_IDS['Movia_S'],
                   OP_IDS['Movia_V']]
OP_IDS['M'] = OP_IDS['Metro']
OP_IDS['F'] = OP_IDS['First']
OP_IDS['D*'] = [OP_IDS['First'], OP_IDS['DSB']]
OP_IDS['D**'] = [OP_IDS['First'],
                 OP_IDS['DSB'],
                 OP_IDS['Stog']]

def _msg(x):

    return f"{x} kwarg accepts int, tuple or list"


def _time_filter(
        timeinfo: np.ndarray,
        val: Union[int, List[int]],
        filt_type: str
        ) -> np.ndarray:
    """
    Filter the time information dataset

    :param timeinfo: time dataset from hdf store
    :type timeinfo: np.ndarray
    :param val: the filter values to use
    :type val: Union[int, List[int]]
    :param filt_type: 'month', 'day', 'hour', 'weekday'
    :type filt_type: str
    :raises TypeError: If filt_type value is an int or a list.
    :return: The filtered array of the time_information dataset
    :rtype: np.ndarray

    """

    col_id = {
        'month': 3,
        'day': 4,
        'hour': 5,
        'weekday': 8
        }

    col = col_id[filt_type]

    if isinstance(val, int):
        timeinfo = timeinfo[timeinfo[:, col] == val]
    elif isinstance(val, (list, tuple)):
        timeinfo = timeinfo[np.isin(timeinfo[:, col], val)]
    else:
        raise TypeError(_msg(filt_type))
    return timeinfo


def _op_intersect(val, ids) -> bool:
    opsets = set(x[1] for x in val)
    wantedset = set.union(*[set(x) for x in ids])

    return wantedset == opsets.intersection(wantedset)

def _start_end(contr_info, startswith=None, endswith=None):
    """filter for kwargs startswith and endswith"""

    if startswith is not None:
        start_ops = OP_IDS[startswith]
        if not isinstance(start_ops, Iterable):
            start_ops = {start_ops}
        contr_info = {k: v for k, v in contr_info.items() if
                      v[0][1] in start_ops}
    if endswith is not None:
        end_ops = OP_IDS[endswith]
        if not isinstance(end_ops, Iterable):
            end_ops = {end_ops}
        contr_info = {k: v for k, v in contr_info.items() if
                      v[-1][1] in end_ops}
    return contr_info

def _contains_stopid():
    return

def _starts_stopid():
    """filter for kwarg starts_at and ends_at"""
    return

def _station_filter(
        stop_info: np.ndarray,
        stations: Union[int, List[int]]
        ) -> np.ndarray:

    if not isinstance(stations, list):
        stations = [stations]
    station_keys = stop_info[np.isin(stop_info[:, 2], stations)][:, 0]

    return stop_info[np.isin(stop_info[:, 0], station_keys)]


class StoreReader():
    """
    A reader class to load rejsekort data.

    Uses h5 and msgpack files from the rejsekort datastores
    """
    # TODO
    def __init__(self, filepath: str, **kwargs) -> None:
        """
        Reader object that reads rejsekort data.

        :param filepath: a string of the filepath to an individual h5 file.
        :type filepath: str
        :param **kwargs: optional keyword arguments
            kwargs : optional keyword arguments
                month : int/list/tuple
                day : int/list/tuple
                hour : int/list/tuple
                weekday : int/list/tuple
                pas_type: defaults to 'main' (all humans and only humans)
                pas_total: int - total number of passengers on the trip
        :type **kwargs: TYPE
        :rtype: None

        """

        self.filepath = filepath
        self.kwargs = kwargs
        self.month = kwargs.get('month', None)
        self.day = kwargs.get('day', None)
        self.hour = kwargs.get('hour', None)
        self.weekday = kwargs.get('weekday', None)
        self.passenger_total = kwargs.get('passenger_total', None)
        self.passenger_type = kwargs.get('passenger_type', 'main')

        # self.region = kwargs.get('region', None)
        self._filter_keys = None
        self.__CACHES = {
            'stop_information': None,
            'time_information': None,
            'price_information': None,
            'passenger_information': None,
            'contractor_information': None
            }

    def __repr__(self):

        return f"Rejsekort StoreReader for {self.filepath}"

    def _load_time(self) -> np.ndarray:
        if self.__CACHES['time_information'] is not None:
            return self.__CACHES['time_information']
        with h5py.File(self.filepath, 'r') as store:
            timeinfo = store['time_information'][:]
        self.__CACHES['time_information'] = timeinfo
        return timeinfo

    def _pas_info(self) -> np.ndarray:
        if self.__CACHES['passenger_information'] is not None:
            return self.__CACHES['passenger_information']
        with h5py.File(self.filepath, 'r') as store:
            pasinfo = store['passenger_information'][:]
        self.__CACHES['passenger_information'] = pasinfo
        return pasinfo

    def _get_time_filtered_keys(self) -> Set[int]:

        timeinfo = self._load_time()

        if hasattr(self, 'month') and self.month is not None:
            timeinfo = _time_filter(timeinfo, self.month, 'month')
        if hasattr(self, 'day') and self.day is not None:
            timeinfo = _time_filter(timeinfo, self.day, 'day')
        if hasattr(self, 'hour') and self.hour is not None:
            timeinfo = _time_filter(timeinfo, self.hour, 'hour')
        if hasattr(self, 'weekday') and self.weekday is not None:
            timeinfo = _time_filter(timeinfo, self.weekday, 'weekday')

        return set(timeinfo[:, 0])

    def _get_operator_filtered_keys(self):
        """Filter on operators."""
        opdata = self._get_contractor_data()
        attrs = ['startswith', 'endswith'] # 'uses',
        if not any(hasattr(self, x) for x in attrs):
            return set(opdata)
        # if hasattr(self, 'uses'):
        #     if hasattr(self, 'bitwise'):
        #         bitwise = self.bitwise
        #         operators = _operator_filter(
        #             opdata, self.operators,
        #             bitwise=bitwise
        #             )
        #     else:
        #         operators = _operator_filter(
        #             opdata, self.operators
        #             )
        # TODO: this is hacking, use properties setter
        if hasattr(self, 'startswith') or hasattr(self, 'endswith'):
            if hasattr(self, 'startswith'):
                startswith = self.startswith
            else:
                startswith = None
            if hasattr(self, 'endswith'):
                endswith = self.endswith
            else:
                endswith = None

            opdata = _start_end(
                opdata, startswith=startswith,
                endswith=endswith
                )

        return set(opdata)

    @staticmethod
    def total_passengers(pas_info: np.ndarray) -> Dict[int, int]:
        """
        Make a dictionary of keys as trip keys and values
        as the total number of passengers on the trip

        :param pas_info: passenger_info dataset
        :type pas_info: np.ndarray
        :return: a dictionary of trip keys and corresponding
            sum of all passengers.
        :rtype: Dict[int, int]

        """

        return {x[0]: sum(x[2:5]) for x in pas_info}

    def _get_pas_filtered_keys(self) -> Set[int]:
        """
        Determine the tripkeys for the passenger filters

        :raises TypeError: if kwarg pas_type is not a list, tuple or string.
        :return: A set of the valid trip keys for the passenger filters.
        :rtype: Set[int]

        """
        pasinfo = self._pas_info()

        pasinfo = pasinfo[
            np.isin(pasinfo[:, 5], PAS_TYPES['main']) |
            np.isin(pasinfo[:, 6], PAS_TYPES['main']) |
            np.isin(pasinfo[:, 7], PAS_TYPES['main'])
            ]
        # trips with dogs
        dog_keys = pasinfo[
            np.isin(pasinfo[:, 6], PAS_TYPES['hund']) |
            np.isin(pasinfo[:, 7], PAS_TYPES['hund'])
            ]
        # trips with bikes
        bike_keys = pasinfo[
            np.isin(pasinfo[:, 6], PAS_TYPES['cykel']) |
            np.isin(pasinfo[:, 7], PAS_TYPES['cykel'])
            ]

        only_humans = pasinfo[
            ~np.isin(pasinfo[:, 0], dog_keys) &
            ~np.isin(pasinfo[:, 0], bike_keys)
            ]

        if hasattr(self, 'passenger_total') and self.passenger_total is not None:
            totals = self.total_passengers(only_humans)
            totals = {k for k, v in totals.items() if v == self.passenger_total}
            only_humans = only_humans[
                np.isin(only_humans[:, 0], list(totals))
                ]

        if hasattr(self, 'passenger_type') and self.passenger_type is not None:
            if isinstance(self.passenger_type, (list, tuple)):
                combo_list = [PAS_TYPES[x] for x in self.passenger_type]
                combo_list = list(chain(*combo_list))
            elif isinstance(self.passenger_type, str):
                combo_list = PAS_TYPES[self.passenger_type]
            else:
                raise TypeError("pas_type kwarg must be a list, tuple or string")

            only_humans = only_humans[
                np.isin(only_humans[:, 5], combo_list) |
                np.isin(only_humans[:, 6], combo_list) |
                np.isin(only_humans[:, 7], combo_list)
                ]
        try:
            return set(only_humans[:, 0])
        except IndexError:
            return set(only_humans)

    def _all_filters(self) -> Set[int]:
        """
        Apply all of the selected filters.

        :return: a set of tripkeys that meet the selected filters
        :rtype: Set[int]

        """

        time_keys = self._get_time_filtered_keys()
        pas_keys = self._get_pas_filtered_keys()
        op_keys = self._get_operator_filtered_keys()
        # geo_keys = self._get_geo_filtered_keys()
        # station_keys = self._get_station_filtered_keys()
        sets = [time_keys, pas_keys, op_keys]
        if op_keys is not None:
            sets.append(op_keys)  # add geo_keys , station_keys
        return set.intersection(*sets)

    def _get_contractor_data(self):
        """
        Find and load the corresponding msgpack file for .h5 file
        :return: the dicrionary of contractor information with tripkeys
            that meet the filter conditions.

        :rtype: Dict

        """
        if self.__CACHES['contractor_information'] is not None:
            return self.__CACHES['contractor_information']

        store_dir = list(pathlib.Path(self.filepath).parents)[1]
        fname = pathlib.Path(self.filepath).parts[-1].split('.')[0]
        mpackname = os.path.join(
            store_dir, 'packs', fname + 'cont.msgpack'
            )
        with open(mpackname, 'rb') as file:
            cont = msgpack.load(file, strict_map_key=False)
        self.__CACHES['contractor_information'] = cont
        return cont

    @staticmethod
    def bad_keys(stop_info: np.ndarray) -> Set[int]:
        """
        Determine the bad tripkeys. Trips that don't have FiCo etc

        :param stop_info: the stop_information dataset in the h5 file.
        :type stop_info: np.ndarray
        :return: a set of tripkeys to discard
        :rtype: Set[int]

        """

        stop_info = stop_info[
            np.lexsort((stop_info[:, 1], stop_info[:, 0]))
            ]

        bad_keys = set()
        for key, group in groupby(stop_info, key=itemgetter(0)):
            models = tuple(x[3] for x in group)
            if not (1 in models and 2 in models and max(models) <= 4):
                bad_keys.add(key)
            if not models[-1] == 2:
                bad_keys.add(key)
            stops = set(x[2] for x in group)
            if 0 in stops:
                bad_keys.add(key)
        return bad_keys

    def _load_dset(self, dset, filter_keys):

        if self.__CACHES[DSET_NAME_MAP[dset]] is not None:
            data = self.__CACHES[DSET_NAME_MAP[dset]]
        else:
            with h5py.File(self.filepath, 'r') as store:
                data = store[dset][:]
                self.__CACHES[DSET_NAME_MAP[dset]] = data
        data = data[np.isin(data[:, 0], filter_keys)]

        return data

    def _flush_kwargs(self):

        kwargs = ['startswith', 'endswith',
                  'remove_bad_trips',
                  'hour', 'weekday', 'month',
                  'passenger_type',
                  'passenger_total']

        self.__dict__ = {
            k: v for k, v in self.__dict__.items() if
            k not in kwargs
            }


    def get_data(self, dset: str, **kwargs) -> Union[np.ndarray, Dict[int, Any]]:
        """
        Get the requested data from the datastore.

        :param dset: the dataset to load from the h5 file.
        :type dset: str
        :param **kwargs: the filter kwargs:
            ['startswith', 'endswith',
             'remove_bad_trips',
             'hour', 'weekday', 'month',
             'passenger_type',
             'passenger_total']

        :type **kwargs:
        :raises ValueError: if the dset argument is not
            one of ['stops', 'price', 'time', 'pas', 'passengers', 'contractor']

        :return: array statisfiying conditions np.ndarray/dict(if dset=='contractor')
        :rtype: TYPE

        """

        dset = dset.lower()
        if dset not in DSET_NAME_MAP:
            raise ValueError(
                f'dset must be in {str(list(DSET_NAME_MAP.keys()))}'
                )
        self._flush_kwargs()
        self.__dict__.update(
            (key, value) for key, value in kwargs.items()
            )

        self._filter_keys = self._all_filters()
        dset = DSET_NAME_MAP[dset]
        if dset != 'contractor_information':
            data = self._load_dset(
                dset, tuple(self._filter_keys)
                )
        else:
            data = self._get_contractor_data()
            data = {k: v for k, v in data.items() if
                    k in self._filter_keys}

        if (hasattr(self, 'remove_bad_trips') and
                self.remove_bad_trips is not None):
            stop_data = self._load_dset(
                'stop_information', tuple(self._filter_keys)
                )
            rm_keys = self.bad_keys(stop_data)
            if dset != 'contractor_information':
                data = data[~np.isin(data[:, 0], list(rm_keys))]
            else:
                data = {k: v for k, v in data.items()
                        if k not in rm_keys}

        return data
