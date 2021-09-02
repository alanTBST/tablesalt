# -*- coding: utf-8 -*-
"""
Classes to interact with passenger stations and the operators that serve them
"""
#standard imports
import json
from itertools import chain, combinations
from typing import Any, AnyStr, Dict, List, Optional, Set, Tuple, Union

import h5py  # type: ignore
import pandas as pd  # type: ignore
import pkg_resources
from tablesalt.common.io import mappers
from tablesalt.topology.tools import determine_takst_region, TakstZones
from tablesalt.topology.stopnetwork import ALTERNATE_STATIONS

M_RANGE: Set[int] = set(range(8603301, 8603400))
S_RANGE: Set[int] = set(range(8690000, 8699999))

MIN_RAIL_UIC: int = 7400000
MAX_RAIL_UIC: int = 9999999

OP_MAP = {v: k.lower() for k, v in mappers['operator_id'].items()}
REV_OP_MAP = {v:k for k, v in OP_MAP.items()}

def _load_default_config() -> Dict[str, str]:
    """load the operator configuration from the package

    :return: a dictionary of network names and operators servicing them
    :rtype: Dict[str, str]
    """
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
    """load the operator configuration from the package

    :return: a dictionary of network lines and operators servicing them
    :rtype: Dict[str, str]
    """

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
    load the operator config file and process it for the given network lines

    :param *lines: a line or lines to load
    :type *lines: str
    :param config_file: path to a config file if not using default, defaults to None
    :type config_file: Optional[AnyStr], optional
    :return: the operator settings
    :rtype: Dict[str, str]

    """

    config_dict = _load_operator_configuration(config_file)

    chosen_lines = {x.lower() for x in lines}

    chosen_operators = set()
    for k, v in config_dict.items():
        if k in chosen_lines:
            try:
                chosen_operators.add(v)
            except TypeError:
                for line in v:
                    chosen_operators.add(config_dict[line])                  
                    chosen_lines.add(line.lower())
                chosen_lines.remove(k) # remove the group name...local, suburban etc
    operator_ids = {
        k.lower(): v for k, v in  mappers['operator_id'].items()
        }

    operator_ids = {
        k: v for k, v in operator_ids.items() if
        k in chosen_operators
        }

    operators = tuple(operator_ids)

    return {
        'operator_ids': operator_ids,
        'operators': operators,
        'config': config_dict,
        'lines': chosen_lines
        }


def load_bus_station_connectors() -> Dict[int, int]:
    """
    Load the bus stops station array from the support data

    :return: a mapping of bus top numbers to station uic numbers
    :rtype: Dict[int, int]

    """
    
    support_store = pkg_resources.resource_filename(
        'tablesalt', 'resources/support_store.h5')

    with h5py.File(support_store, 'r') as store:
        bus_map = store['datasets/bus_closest_station'][:]

    return {x[0]: x[1] for x in bus_map}


def _load_bus_station_map() -> Dict[int, int]:
    filepath = pkg_resources.resource_filename(
        'tablesalt', 'resources/bus_closest_station.json')

    with open(filepath, 'r') as f:
        bus_map = json.load(f)

    return {int(bstop): station for bstop, station in bus_map.items()}


def _alternate_stop_map():

    max_alternates = max(len(x) for x in ALTERNATE_STATIONS.values())
    alternate_dicts = {x: {} for x in range(max_alternates)}
    for k, v in ALTERNATE_STATIONS.items():
        for i, stopid in enumerate(v):
            alternate_dicts[i][k] = stopid

    return alternate_dicts

def _load_default_passenger_stations(*lines: str) -> pd.core.frame.DataFrame:

    """load the passenger stations data

    :return: a dataframe of stations and operators used to query
    :rtype: pd.core.frame.DataFrame
    """
    fp = pkg_resources.resource_filename(
            'tablesalt',
            'resources/networktopodk/operator_stations.csv'
            )

    pas_stations = pd.read_csv(
        fp, encoding='iso-8859-1'
        )
    pas_stations.columns = [x.lower() for x in pas_stations.columns]

    alternates = pas_stations.query("stop_id in @ALTERNATE_STATIONS")

    alternate_dicts = _alternate_stop_map()

    new_frames = []
    for i, d in alternate_dicts.items():
        new_frame = alternates.copy()
        new_frame = new_frame.query("stop_id in @d")
        new_frame.loc[:, 'stop_id'] = new_frame.loc[:, 'stop_id'].map(d)
        new_frames.append(new_frame)

    concat_new_frames = pd.concat(new_frames, ignore_index=True)
    out = pd.concat([pas_stations, concat_new_frames], ignore_index=True)
    cols = ['stop_id'] + list(lines)
    out = out[cols]
    return out

# make a separate lookup class
class StationOperators():

    BUS_ID_MAP: Dict[int, int] = _load_bus_station_map()

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
        #self._station_dict_set = {
        #    k: tuple(v) for k, v in self._station_dict.items()
        #    }
        self._stop_zone_map = TakstZones().stop_zone_map()
        
        #self._create_lookups()

    def _make_query(self, intup: Tuple[str, ...]) -> str:
        """
        create the query to select valid values from the dataframe
        """
        lines = self._settings['lines']
        qry = []
        for oper in lines:
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
        lines = self._settings['lines']
        pas_stations = _load_default_passenger_stations(*lines)
        # range(1, ..) - include single operators in the permutations

        all_perms = [
            x for l in range(1, len(lines)) for
            x in combinations(lines, l)
            ]

        out = {}
        for perm_tup in all_perms:
            out[perm_tup] = pas_stations.query(
                self._make_query(perm_tup)
                )['stop_id'].tolist()

        out = {k: v for k, v in out.items() if v}

        x = zip(out.keys(), out.values())
        x = sorted(x, key=lambda x: x[0])
        outdict = {}
        seen = set()
        for k, v in x:
            val = tuple(set(k)), tuple(set(v))
            if val not in seen:
                outdict[k] = v
                seen.add(val)
        
        return  {frozenset(k): v for k, v in outdict.items()}

    def _station_type(self, stop_number: int) -> Tuple[str, ...]:
        """
        return the operator key if the given stop_number
        is in the values
        """
        for k, v in self._station_dict_set.items():
            if stop_number in v:
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

    def _get_operator(self, stop_number: int) -> Tuple[str, ...]:
        """return the operators that survice the station

        :param stop_number: the stop uic number
        :type stop_number: int
        :raises KeyError: if the station/stop is not found
        :return:  a tuple of operators that service the stop
        :rtype: Tuple[str, ...]
        """
        try:
            return self._lookup[stop_number]
        except KeyError:
            if stop_number > MAX_RAIL_UIC or stop_number < MIN_RAIL_UIC:
                return tuple((self._settings['config']['bus'], ))
        raise KeyError("stop_number not found")

    def _get_line(self, stop_number: int) -> Tuple[str, ...]:
        """return the line names that the station is on

        :param stop_number: the stop uic number
        :type stop_number: int
        :raises KeyError: the stop uic number
        :return: a tuple of line names that the stop is on
        :rtype: Tuple[str, ...]
        """
        minidict = {
            v: k for k, v in self._settings['config'].items()
            if k in self.lines
            }
        try:
            return tuple(minidict[x] for x in self._lookup[stop_number])
        except KeyError:
             if stop_number > MAX_RAIL_UIC or stop_number < MIN_RAIL_UIC:
                return tuple(('bus', ))
        raise KeyError("stop_number not found")

    def _get_operator_id(self, stop_number: int) -> Tuple[int, ...]:
        """[summary]

        :param stop_number: the stop uic number
        :type stop_number: int
        :raises KeyError: the stop uic number
        :return:  a tuple of operator ids that the stop is on
        :rtype: Tuple[int, ...]
        """
        try:
            # 1 is movia_H, this must be made generic
            op_name = determine_takst_region(self._stop_zone_map[stop_number])
            return tuple(
                self._settings['operator_ids'].get(x, REV_OP_MAP[op_name]) for x in self._lookup[stop_number]
                )
        except KeyError:
            try:
                if stop_number > MAX_RAIL_UIC or stop_number < MIN_RAIL_UIC:
                    op_name = determine_takst_region(self._stop_zone_map[stop_number])
                    return tuple((REV_OP_MAP[op_name], ))
            except KeyError:
                raise KeyError(f"stop number = {stop_number} is not a valid stop point")
        raise KeyError(f"stop_number = {stop_number}  not found")


    def get_ops(
        self, stop_number: int,
        format: Optional[str] = 'operator_id'
        ) -> Tuple[Union[int, str], ...]:
        """
        Returns a tuple of the operators at the given station id

        :param stop_number: uic number of the station
        :type stop_number: int
        :param format: 'operator_id', 'operator' or 'line', defaults to 'operator_id'
            'operator' - returns str values representing the operators at the stop
            'line' - returns line names of the stop

        :type format: Optional[str], optional
        :raises ValueError: if incorrect format is given
        :return: the operators or lines serving the given station
        :rtype: Tuple[Union[int, str], ...]

        :Example:

        to return the operators at Copenhagen central station:
            the uic number is 8600626

        >>> op_getter = StationOperators()
        >>> cph_operator_ids = op_getter.get_ops(8600626, format='operator_id')
        >>> cph_operator_ids
        >>> (4, 8, 5, 6)
        >>> cph_operators = op_getter.get_ops(8600626, format='operator')
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
            'operator_id': self._get_operator_id(stop_number),
            'operator': self._get_operator(stop_number),
            'line': self._get_line(stop_number)
            }
        return fdict[format]


    def _check_bus_location(self, bus_stop_id: int) -> int:
        """map a bus id to a station if possible, else 0

        :param bus_stop_id: the stop number
        :type bus_stop_id: int
        :return: the mapped station number or 0 if no mapping exists
        :rtype: int
        """

        return self.BUS_ID_MAP.get(bus_stop_id, 0)

    def _pair_operator(
        self, start_uic: int,
        start_ops: Tuple[str, ...],
        end_ops: Tuple[str, ...]
        ) -> Tuple[str, ...]:

        """return the possible operator strings that can service the
        a start and end point

        :return: a tuple of operators that serve the station pair
        :rtype: Tuple[str, ...]
        """

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
        """return the possibe operator ids that can service
        a start and end point

        :param start_uic: station id of start
        :type start_uic: int
        :param start_ops: station operators of start
        :type start_ops: Tuple[int, ...]
        :param end_ops: station operators of end
        :type end_ops: Tuple[int, ...]
        :return: a tuple of the possible operator ids
        :rtype: Tuple[int, ...]
        """


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
        """return the possible lines that can service
        a start and end point

        :param start_uic: the stop number of
        :type start_uic: int
        :param start_ops: the possible operators at the start
        :type start_ops: Tuple[str, ...]
        :param end_ops: the possible operators at the end
        :type end_ops: Tuple[str, ...]
        :return: the possible lines
        :rtype: Tuple[str, ...]
        """

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

        :Example:

        to return the possbible operators servicing a leg from copenhagen central
        station to helsingør station

        >>> op_getter = StationOperators('kystbanen', 'suburban', 'fjernregional', 'local', 'metro')
        >>> op_getter.station_pair(8600626, 8600669, format='line')
            ('kystbanen',)
        >>> opgetter.station_pair(8600626, 8600669, format='operator')
            ('first',)
        >>> opgetter.station_pair(8600626, 8600669, format='operator_id')
            (8,)
        """

        try:
            return self.ID_CACHE[(start_uic, end_uic, format)]
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
            self.ID_CACHE[(start_uic, end_uic, format)] = returnval
        return returnval
