# -*- coding: utf-8 -*-
"""
Classes to interact with passenger stations and the operators that serve them
"""
#standard imports
import json
from collections import defaultdict
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

    use_groups = False # if a group of lines is input
    chosen_operators = set()
    for k, v in config_dict.items():
        if k in chosen_lines:
            try:
                chosen_operators.add(v)
            except TypeError:
                use_groups = True
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
        'lines': chosen_lines,
        'use_groups': use_groups
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

    metromap = mappers['metro_map']
    revmetromap = mappers['metro_map_rev']
    
    sstops = mappers['s_uic']
    mstops = mappers['m_uic']

    sstopsdict = {}
    for stopid in sstops:
        normalid = stopid - 90000
        ms = revmetromap.get(normalid)
        alts = [d.get(normalid) for d in alternate_dicts.values()]
        alts = [x for x in alts if x and x != stopid]
        if ms:
            alts = alts + [ms]
        for x in alts:
            sstopsdict[x] = stopid
    
    mstopsdict = {}
    for stopid in mstops:
        ss = metromap.get(stopid)
        alts = [d.get(ss) for d in alternate_dicts.values()]
        alts = [x for x in alts if x and x != stopid]
        for x in alts:
            mstopsdict[x] = stopid

    alternate_dicts[max_alternates] = metromap
    alternate_dicts[max_alternates+1] = revmetromap

    return alternate_dicts, sstopsdict, mstopsdict

def _load_default_passenger_stations() -> pd.core.frame.DataFrame:

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

    alternate_dicts, sstopsdict, mstopsdict = _alternate_stop_map()

    frames = []
    for i, d in alternate_dicts.items():

        new_frame = pas_stations.copy()
        new_frame = new_frame.query("stop_id in @d")
        new_frame.loc[:, 'stop_id'] = new_frame.loc[:, 'stop_id'].map(d)
        
        sframe = new_frame.query("stop_id in @sstopsdict").copy()
        sframe.loc[:, 'stop_id'] = sframe.loc[:, 'stop_id'].map(sstopsdict)

        mframe = new_frame.query("stop_id in @mstopsdict").copy()
        mframe.loc[:, 'stop_id'] = mframe.loc[:, 'stop_id'].map(mstopsdict)        
        
        new_frame = new_frame.set_index('stop_id')
        sframe = sframe.set_index('stop_id')
        mframe = mframe.set_index('stop_id')
        
        frames.append(new_frame)
        frames.append(sframe)
        frames.append(mframe)



    pas_stations = pas_stations.query("stop_id > @MIN_RAIL_UIC ")
    pas_stations = pas_stations.set_index('stop_id')
    frames.append(pas_stations)

    return frames


def _grouped_lines_dict(config_dict):

    groupdict = {}
    for line, info in config_dict.items():
        if isinstance(info, list):
            for l in info:
                groupdict[l] = line
    return groupdict


# make a separate lookup class
class StationOperators():

    BUS_ID_MAP: Dict[int, int] = _load_bus_station_map()

    LINE_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    OP_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}

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
        stop_lookup, line_lookup = self._pas_station_dict()   
        self._stop_lookup = stop_lookup   
        self._line_lookup = line_lookup   
        self._stop_zone_map = TakstZones().stop_zone_map()
        self._group_lines = _grouped_lines_dict(self._settings['config'])

    def _pas_station_dict(self) -> Dict[Tuple[str, ...], List[int]]:
        """
        create the station dictionary from the underlying
        dataset
        """
        lines = self._settings['lines']
        pas_stations = _load_default_passenger_stations()
  
        line_lookup = defaultdict(set)
        for line in lines:
            for frame in pas_stations:
                s = frame[line]
                good = set(s[s > 0].index)
                line_lookup[line].update(good)
        
        stop_lookup = defaultdict(set)
        for k, v in line_lookup.items():
            for stopid in v:
                stop_lookup[stopid].add(k)
        
        return stop_lookup, line_lookup

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

    def station_pair(
        self,
        start_uic: int,
        end_uic: int,
        format: Optional[str] = 'operator'
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
        """ 
        if format == 'operator':
            try:
                return self.OP_CACHE[(start_uic, end_uic)]
            except KeyError:
                pass
        elif format == 'line':
            try:
                return self.LINE_CACHE[(start_uic, end_uic)]
            except KeyError:
                pass
        else:
            raise ValueError(f"format={format} not available")
        
        start_bus = start_uic > MAX_RAIL_UIC or start_uic < MIN_RAIL_UIC
        end_bus = end_uic > MAX_RAIL_UIC or end_uic < MIN_RAIL_UIC

        if start_bus:
            startzone = self._stop_zone_map.get(start_uic)
            if not startzone:
                raise ValueError(f"Unknown bus stop id={start_uic}")
            region = determine_takst_region(startzone)
            line = {f'{region}_bus'}
            operator = {self._settings['config']['bus'][region]}
            
            self.OP_CACHE[(start_uic, end_uic)] = operator
            self.LINE_CACHE[(start_uic, end_uic)] = line

                
        if not start_bus:
            start_lines = self._stop_lookup[start_uic]
        if not end_bus:
            end_lines = self._stop_lookup[end_uic]
        else:
            bus_loc_check = self._check_bus_location(end_uic)
            if not bus_loc_check:
                 raise ValueError(f"Unknown bus stop id={end_uic}")
            end_lines = self._stop_lookup[bus_loc_check]
                       
        line_intersection = start_lines.intersection(end_lines)
        
        if not line_intersection:
            # find missing stop point
            start_groups = {self._group_lines.get(x) for x in start_lines}
            start_groups = {x for x in start_groups if x}
            end_groups = {self._group_lines.get(x) for x in end_lines}  
            end_groups = {x for x in end_groups if x}
            group_intersection = start_groups.intersection(end_groups)

            if group_intersection:
                group_lines, possible_operators = self._has_grp_intersection(
                    group_intersection
                    )
                self.OP_CACHE[(start_uic, end_uic)]  = possible_operators
                self.LINE_CACHE[(start_uic, end_uic)] = group_lines
            else: 
                self.OP_CACHE[(start_uic, end_uic)]  = set()
                self.LINE_CACHE[(start_uic, end_uic)] = set()
        else:                      
            possible_operators, possible_lines = self._has_line_intersection(
                start_uic, line_intersection
                )

            self.OP_CACHE[(start_uic, end_uic)] = possible_operators
            self.LINE_CACHE[(start_uic, end_uic)] = possible_lines                      
              
        if format == 'operator':          
            return self.OP_CACHE[(start_uic, end_uic)]
        return self.LINE_CACHE[(start_uic, end_uic)]

    def _has_grp_intersection(self, group_intersection):
        
        if len(group_intersection) == 1:
            group = group_intersection.pop()
            group_lines = set(self._settings['config'][group])
            possible_operators = {self._settings['config'][x] for x in group_lines}
                
        else:
            group_lines = set()
            possible_operators = set()
        
        return group_lines, possible_operators

    
    def _has_line_intersection(self, start_uic, line_intersection):

        if start_uic in mappers['s_uic']:
            possible_lines = {x for x in line_intersection if 
                              x in self._settings['config']['suburban']}
            possible_operators = {
                self._settings['config'][x] for x in possible_lines
                }
        else:
            possible_lines = {x for x in line_intersection if 
                              x not in self._settings['config']['suburban']}
            possible_operators = {
                self._settings['config'][x] for x in possible_lines
                }
        return possible_operators, possible_lines