# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:38:57 2019

@author: alkj

"""
#standard imports
import json
import pkg_resources
from itertools import chain, permutations
from typing import (
    Any,
    AnyStr,
    Dict,
    Optional,
    Tuple,
    List,
    Union,
    Set
    )

import h5py #type: ignore
import pandas as pd #type: ignore


from tablesalt.common.io import mappers

M_RANGE: Set[int] = set(range(8603301, 8603400))
S_RANGE: Set[int] = set(range(8690000, 8699999))

MIN_RAIL_UIC: int = 7400000
MAX_RAIL_UIC: int = 9999999


def _load_default_config() -> Dict[str, str]:
    fp = pkg_resources.resource_filename(
    'tablesalt', 'resources/config/operator_config.json'
    )

    config_dict: Dict[str, str]

    with open(fp, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return config_dict


def _load_operator_configuration(
    config_file: Optional[AnyStr] = None
    ) -> Dict[str, str]:


    if config_file is None:
        config_dict = _load_default_config()
    else:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
    return config_dict

def _load_operator_settings(
    *lines: str,
    config_file: Optional[AnyStr] = None
    ) -> Dict[str, str]:
    """

    :param *lines: DESCRIPTION
    :type *lines: str
    :param config_file: DESCRIPTION, defaults to None
    :type config_file: Optional[AnyStr], optional
    :return: DESCRIPTION
    :rtype: Dict[str, str]

    """

    config_dict = _load_operator_configuration(config_file)

    chosen_lines = {x.lower() for x in lines}
    chosen_operators = list(
        set(v for k, v in config_dict.items() if k in chosen_lines)
    )

    operator_ids = {
        k.lower(): v for k, v in  mappers['operator_id'].items()
        }

    operator_ids = {
        k: v for k, v in operator_ids.items() if
        k in chosen_operators
        }

    operators = tuple(set(config_dict[x] for x in chosen_lines))

    return {
        'operator_ids': operator_ids,
        'operators': operators,
        'config': config_dict
        }


def load_bus_station_connectors() -> Dict[int, int]:
    """
    Load the bus stops station array from the support data
    :return: DESCRIPTION
    :rtype: Dict[int, int]

    """
    support_store = pkg_resources.resource_filename(
        'tablesalt', 'resources/support_store.h5')

    with h5py.File(support_store, 'r') as store:
        bus_map = store['datasets/bus_closest_station'][:]

    return {x[0]: x[1] for x in bus_map}

def _load_default_passenger_stations(*lines: str) -> pd.core.frame.DataFrame:


    fp = pkg_resources.resource_filename(
            'tablesalt',
            'resources/networktopodk/operator_stations.csv'
            )

    pas_stations = pd.read_csv(
        fp, encoding='utf-8'
        )

    pas_stations.columns = [x.lower() for x in pas_stations.columns]

    base_cols = [
        'uic', 'parent_uic', 'stationsnavn',
        'forkortelse', 'region nr', 'region navn',
        'kommune nr', 'kommune navn'
        ]
    line_cols = [x for x in pas_stations.columns if x in lines]

    cols = base_cols + line_cols
    pas_stations = pas_stations[cols]
    return pas_stations



class StationOperators():

    BUS_ID_MAP: Dict[int, int] = load_bus_station_connectors()

    ID_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def __init__(self, *lines: str) -> None:
        """
        Class used to find the operators at stations and the
        interoperability at stations

        :param *lines: the physical rail lines to include
            options: any combination of:
                        'kystbanen',
                        'local',
                        'metro',
                        'suburban',
                        'fjernregional'

        :type *lines: str
        :return: ''
        :rtype: None

        """


        self.lines = lines
        self._settings = _load_operator_settings(*self.lines)
        self._station_dict = self._pas_station_dict()
        self._station_dict_set = {
            k: tuple(v) for k, v in self._station_dict.items()
            }
        self._create_lookups()

    def _make_query(self, intup: Tuple[str, ...]) -> str:
        """
        create the query to select valid values from the dataframe
        """

        qry = []
        for oper in self.lines:
            if 'bus' not in oper:
                val = int(oper in intup)
                string = f"{oper} == {val}"
                qry.append(string)

        full_qry = " & ".join(qry)

        return full_qry


    def _pas_station_dict(self) -> Dict[Tuple[str, ...], List[int]]:
        """
        create the station dictionary from the underlying
        dataset
        """

        pas_stations = _load_default_passenger_stations(*self.lines)
        s_exceptions = pas_stations.query(
            "uic > 8690000")['parent_uic'].dropna().astype(int).tolist()

        # range(1, ..) - include single operators in the permutations

        all_perms = [
            x for l in range(1, len(self.lines)) for
            x in permutations(self.lines, l)
            ]

        out = {}
        for perm_tup in all_perms:
            out[perm_tup] = pas_stations.query(
                self._make_query(perm_tup)
                )['uic'].tolist()

        out = {k: v for k, v in out.items() if v}

        config = self._settings['config']
        for k, v in out.items():
            if k[0] == config["metro"]:
                out[k] = [x for x in v if x in M_RANGE]
            else:
                out[k] = [x for x in v if x not in M_RANGE]
        for k, v in out.items():
            if k[0] in (
                    config["fjernregional"],
                    config["kystbanen"],
                    config["local"]
                    ):
                out[k] = [x for x in v if x not in S_RANGE]
#            else:
#                out[k] = [x for x in v if x in S_RANGE]
        for k, v in out.items():
            if k[0] == config["suburban"] and len(k) > 1:
                out[k] = [
                    x for x in v if x in S_RANGE or x in s_exceptions
                    ]

        x = zip(out.keys(), out.values())
        x = sorted(x, key=lambda x: x[0])
        outdict = {}
        seen = set()
        for k, v in x:
            val = tuple(set(k)), tuple(set(v))
            if val not in seen:
                outdict[k] = v
                seen.add(val)
        return outdict

    def _station_type(self, uic_number: int) -> Tuple[str, ...]:
        """
        return the operator key if the given uic_number
        is in the values
        """
        for k, v in self._station_dict_set.items():
            if uic_number in v:
                return k
        raise ValueError("station number not found")

    def _create_lookups(self) -> None:
        """
        create the dictionaries that are used in the public
        methods
        """

        all_stations = set(chain(*self._station_dict_set.values()))
        self._lookup_name = {x: self._station_type(x) for x in all_stations}
        self._lookup = {
            k: tuple(self._settings['config'][x] for x in v) for
            k, v in self._lookup_name.items()
            }

    def _get_operator(self, uic_number: int) -> Tuple[str, ...]:
        try:
            return self._lookup[uic_number]
        except KeyError:
            if uic_number > MAX_RAIL_UIC or uic_number < MIN_RAIL_UIC:
                return tuple((self._settings['config']['bus'], ))
        raise KeyError("uic number not found")

    def _get_line(self, uic_number: int) -> Tuple[str, ...]:

        minidict = {
            v: k for k, v in self._settings['config'].items()
            if k in self.lines
            }
        try:
            return tuple(minidict[x] for x in self._lookup[uic_number])
        except KeyError:
             if uic_number > MAX_RAIL_UIC or uic_number < MIN_RAIL_UIC:
                return tuple(('bus', ))
        raise KeyError("uic number not found")

    def _get_operator_id(self, uic_number: int) -> Tuple[int, ...]:
        try:
            # 1 is movia_H
            return tuple(
                self._settings['operator_ids'].get(x, 1) for x in self._lookup[uic_number]
                )
        except KeyError:
            if uic_number > MAX_RAIL_UIC or uic_number < MIN_RAIL_UIC:
                return tuple((1, )) # 0 index gives 1 for movia_H..all sjælland
        raise KeyError("uic number not found")


    def get_ops(
        self, uic_number: int,
        format: Optional[str] = 'operator_id'
        ) -> Tuple[Union[int, str], ...]:
        """
        Returns a tuple of the operators at the given station id

        :param uic_number: uic number of the station
        :type uic_number: int
        :param format: 'operator_id', 'operator' or 'line', defaults to 'operator_id'
            'operator' - returns str values representing the operators at the stop
            'line' - returns line names of the stop

        :type format: Optional[str], optional
        :raises ValueError: if incorrect format is given
        :return: the operators or lines serving the given station
        :rtype: Tuple[Union[int, str], ...]

        :Example:
        ----------
        to return the operators at Copenhagen central station:
            the uic number is 8600626

        >>> op_getter = StationOperators()
        >>> cph_operator_ids = op_getter.get_ops(8600626, format='operator_id')
        >>> cph_operator_ids
        >>> (4, 8, 5, 6)
        >>> cph_operators = op_getter.get_ops(86000626, format='operator')
        >>> cph_operators
        >>> ('first', 's-tog', 'dsb', 'metro')

        Copenhagen central also has an s-tog platform that has
        the uic number 8690626

        in this case:

        >>> cph_stog_operators = op_getter.get_ops(8690626, format='operator')
        >>> cph_stog_operators
        >>> ('s-tog', 'first', 'dsb', 'metro')

        """

        if format not in {'operator', 'operator_id', 'line'}:
            raise ValueError(
        "format must be one of 'operator', 'operator_id', 'line'"
        )

        fdict: Dict[str,  Union[Tuple[int, ...], Tuple[str, ...]]]
        fdict = {
            'operator_id': self._get_operator_id(uic_number),
            'operator': self._get_operator(uic_number),
            'line': self._get_line(uic_number)
            }
        return fdict[format]


    def _check_bus_location(self, bus_stop_id) -> int:

        return self.BUS_ID_MAP.get(bus_stop_id, 0)

    def _pair_operator(
        self, start_uic: int,
        start_ops: Tuple[str, ...],
        end_ops: Tuple[str, ...]
        ) -> Tuple[str, ...]:

        intersect = set(start_ops).intersection(set(end_ops))
        if len(intersect) == 1:
            return tuple(intersect)

        if intersect:
            if start_uic in M_RANGE:
                return tuple((self._settings['config']['metro'], ))

            if start_uic in S_RANGE:
                return tuple((self._settings['config']['suburban'], ))

        return tuple(intersect)

    def _pair_operator_id(
        self, start_uic: int,
        start_ops: Tuple[int, ...],
        end_ops: Tuple[int, ...]
        ) -> Tuple[int, ...]:


        intersect = set(start_ops).intersection(set(end_ops))
        if len(intersect) == 1:
            return tuple(intersect)

        if intersect:

            if start_uic in M_RANGE:
                return tuple(
                    (self._settings['operator_ids'][self._settings['config']['metro']], )
                    )
            if start_uic in S_RANGE:
                return tuple(
                    (self._settings['operator_ids'][self._settings['config']['suburban']], )
                    )

        return tuple(intersect)


    def _pair_line(
        self, start_uic: int,
        start_ops: Tuple[str, ...],
        end_ops: Tuple[str, ...]
        ) -> Tuple[str, ...]:

        intersect = set(start_ops).intersection(set(end_ops))
        if len(intersect) == 1:
            return tuple(intersect)
        if intersect:
            if start_uic in M_RANGE:
                return tuple(('metro', ))
            if start_uic in S_RANGE:
                return tuple(('suburban', ))

        return tuple(intersect)


    def station_pair(
        self,
        start_uic: int,
        end_uic: int,
        format: Optional[str] = 'operator_id'
        ) -> Union[Tuple[int, ...], Tuple[str, ...]]:
        """
        Returns the possible operators that can perform
        the journey between a given station pair

        :param start_uic: the uic number of the start station
        :type start_uic: int
        :param end_uic: the uic number of the end station
        :type end_uic: int
        :param format:  the way in which you desire the output, defaults to 'operator_id'
            'operator_id' - returns values as integer ids representing operators
            'operator' - returns values as strings of the operator name

        :type format: Optional[str], optional
        :raises ValueError: if the end stopid is not mappable to a station
        :return: the operators serving the given station pair
        :rtype: Tuple[int, ...]

        """

        try:
            return self.ID_CACHE[(start_uic, end_uic)]
        except KeyError:
            pass

        start_bus = start_uic > MAX_RAIL_UIC or start_uic < MIN_RAIL_UIC
        end_bus = end_uic > MAX_RAIL_UIC or end_uic < MIN_RAIL_UIC

        if start_bus:
            return self.get_ops(start_uic, format=format)

        if not start_bus and end_bus:
            bus_loc_check = self._check_bus_location(end_uic)
            if bus_loc_check:
                end_uic = bus_loc_check
            else:
                raise ValueError(
            f"Can't find station for end of leg bus stop id {end_uic}"
            )
        start_ops = self.get_ops(start_uic, format=format)
        end_ops = self.get_ops(end_uic, format=format)

        fdict: Dict[str,  Union[Tuple[int, ...], Tuple[str, ...]]]

        fdict = {
            'operator_id': self._pair_operator_id(start_uic, start_ops, end_ops),
            'operator': self._pair_operator(start_uic, start_ops, end_ops),
            'line': self._pair_line(start_uic, start_ops, end_ops)
            }

        returnval: Union[Tuple[int, ...], Tuple[str, ...]]
        returnval = fdict[format]
        if format == 'operator_id':
            self.ID_CACHE[(start_uic, end_uic)] = returnval
        return returnval


