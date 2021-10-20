# -*- coding: utf-8 -*-
"""
Classes to interact with passenger stations and the operators that serve them
"""
#standard imports
import ast
import json
from collections import defaultdict
from itertools import chain, permutations
from typing import Any, AnyStr, Dict, List, Optional, Set, Tuple, Union

import pandas as pd  # type: ignore
import pkg_resources
from tablesalt import transitfeed
from tablesalt.common.io import mappers
from tablesalt.resources.config.config import load_config
from tablesalt.topology.stopnetwork import ALTERNATE_STATIONS
from tablesalt.topology.tools import TakstZones, determine_takst_region
from tablesalt.transitfeed.feed import TransitFeed

CONFIG = load_config()

SUBURBAN_UIC = dict(CONFIG['suburban_platform_numbers'])
SUBURBAN_UIC = {int(k): int(v) for k, v in SUBURBAN_UIC.items()}

METRO_UIC = dict(CONFIG['metro_platform_numbers'])
METRO_UIC = {int(k): int(v) for k, v in METRO_UIC.items()}
REV_METRO_UIC = {v:k for k, v in METRO_UIC.items()}

LOCAL_UIC_1 =  dict(CONFIG['local_platform_numbers_1'])
LOCAL_UIC_1 = {int(k): int(v) for k, v in LOCAL_UIC_1.items()}

LOCAL_UIC_2 =  dict(CONFIG['local_platform_numbers_2'])
LOCAL_UIC_2 = {int(k): int(v) for k, v in LOCAL_UIC_2.items()}

LOCAL_UIC = {**LOCAL_UIC_1, **LOCAL_UIC_2}

ALTERNATE_UIC =  dict(CONFIG['alternate_numbers'])
ALTERNATE_UIC = {int(k): ast.literal_eval(v) for k, v in ALTERNATE_UIC.items()}


SUBURBAN_TEST_STOPS = set(ast.literal_eval(CONFIG['suburban_farum_cph']['test_stops']))
METRO_TEST_STOPS = set(ast.literal_eval(CONFIG['metro_lindevang']['test_stops']))
KASTRUP_TEST_STOPS =  set(ast.literal_eval(CONFIG['kastrup_cph']['test_stops']))
LOCAL_TEST_STOPS =  set(ast.literal_eval(CONFIG['local']['test_stops']))


M_RANGE: Set[int] = set(range(8603301, 8603400))
S_RANGE: Set[int] = set(range(8690000, 8699999))

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

def _load_default_passenger_stations() -> List[pd.core.frame.DataFrame]:

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

feed = transitfeed.archived_transitfeed('20211011_20220105')
class StationOperators:

    BUS_ID_MAP: Dict[int, int] = _load_bus_station_map()

    def __init__(self, feed: TransitFeed) -> None:
        """class to determine the operator between stop point ids

        :param feed: a gtfs transitfeed from rejseplanen
        :type feed: TransitFeed
        """

        self.feed = feed       
        self._suburban_operator, self._metro_operator, self._local_operator = \
            self._determine_operators()

        self._lookup = self._process_stop_times()
        self._suburban_lookup = self._process_suburban_stops()
    
    def _determine_operators(self):

        stoptimes = list(self.feed.stop_times.data.items())
        
        suburban_trip = None
        metro_trip = None
        kastrup_trip = None
        local_trip = None  # NOTE. only one local operator
        
        for tripid, stoptime in stoptimes:
            stopids = set(x['stop_id'] for x in stoptime)
            if stopids.intersection(SUBURBAN_TEST_STOPS):
                suburban_trip = tripid
            
            elif stopids.intersection(METRO_TEST_STOPS):
                metro_trip = tripid
            elif stopids.intersection(KASTRUP_TEST_STOPS):
                kastrup_trip = tripid     
            elif stopids.intersection(LOCAL_TEST_STOPS):
                local_trip = tripid 
                    

            if (suburban_trip is not None and 
                metro_trip is not None and
                kastrup_trip is not None and 
                local_trip is not None):
                break        
        else:
            raise ValueError("Cannot determine suburban/metro operator")
    
        sub_agency_name = self.feed.get_agency_name_for_trip(suburban_trip)
        met_agency_name = self.feed.get_agency_name_for_trip(metro_trip)
        kast_agency_name = self.feed.get_agency_name_for_trip(kastrup_trip)
        local_agency_name = self.feed.get_agency_name_for_trip(local_trip)
        return sub_agency_name, met_agency_name, local_agency_name

    def _process_stop_times(self) -> Dict[int, Tuple[Tuple[int, int]]]:
        relation_operators = {}
        
        for trip_id, stoptime in self.feed.stop_times.data.items():
            
            stopids = (x['stop_id'] for x in stoptime)
            perms = set(permutations(stopids, 2))
            
            route_id = self.feed.trips.get(trip_id)['route_id']
            agency_id = self.feed.routes.get(route_id)['agency_id']
            agency = self.feed.agency.get(agency_id)            
            
            for stop_relation in perms:
                try:
                    relation_operators[stop_relation].add(agency)      
                except KeyError:
                    relation_operators[stop_relation] = {agency}

            # put in SUBURBAN_UIC
            if agency == self._suburban_operator:
                start_suburban_perms = {(SUBURBAN_UIC.get(x[0], x[0]), x[1]) for x in perms}
                # end_metro_perms
                end_metro_perms = {(x[0], METRO_UIC.get(x[1], x[1])) for x in perms}
                # end_local_perms
                end_local_perms = {(x[0], LOCAL_UIC.get(x[1], x[1])) for x in perms}
                suburban_perms = start_suburban_perms - perms
                local_perms = end_local_perms - perms
                metro_perms = end_metro_perms - perms
                suburban_ops = {stop_relation: {self._suburban_operator} for stop_relation in suburban_perms}  
                local_ops = {stop_relation: {self._suburban_operator} for stop_relation in local_perms}    
                metro_ops = {stop_relation: {self._suburban_operator} for stop_relation in metro_perms}      
                relation_operators.update(suburban_ops)
                relation_operators.update(local_ops)
                relation_operators.update(metro_ops)
            # put in LOCAL
            elif agency == self._local_operator:
                start_local_perms = {(LOCAL_UIC.get(x[0], x[0]), x[1]) for x in perms}
                local_perms = start_local_perms - perms
                local_ops = {stop_relation: {self._local_operator} for stop_relation in local_perms}
                relation_operators.update(local_ops)
                # put in METRO
            elif agency == self._metro_operator:
                start_metro_perms = {(x[0], REV_METRO_UIC.get(x[1], x[1])) for x in perms}
                metro_perms = start_metro_perms - perms
                metro_ops = {
                    stop_relation: {self._metro_operator} for 
                    stop_relation in metro_perms
                    }
                relation_operators.update(metro_ops)
           
        
        return relation_operators
 
    def station_pair(self, start_stop_id: int, end_stop_id: int) -> Set[str]:

        try:
            return self._lookup[(start_stop_id, end_stop_id)]          
        except KeyError:
            op = self._lookup[(SUBURBAN_UIC.get(start_stop_id, start_stop_id), end_stop_id)]
            return op.intersection({self._suburban_operator})
        except KeyError:
            op = self._lookup[(SUBURBAN_UIC.get(start_stop_id, start_stop_id), SUBURBAN_UIC.get(end_stop_id, end_stop_id))] 
            return op.intersection({self._suburban_operator})       
        except KeyError:
            op = self._lookup[(start_stop_id, SUBURBAN_UIC.get(end_stop_id, end_stop_id))] 
            return op - {self._suburban_operator}          
        except KeyError:
            raise

    

# make a separate lookup class
# class StationOperators():

#     BUS_ID_MAP: Dict[int, int] = _load_bus_station_map()

#     LINE_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}
#     OP_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}

#     def __init__(self, *lines: str) -> None:
#         """
#         Class used to find the operators at stations and the
#         interoperability at stations

#         :param *lines: the physical rail lines to include
#             options: any combination of:
#                         'kystbanen',
#                         'local',
#                         'metro',
#                         'suburban',
#                         'fjernregional'

#         :type *lines: str
#         :return: ''
#         :rtype: None

#         """


#         self.lines = lines
#         self._settings = _load_operator_settings(*self.lines)
#         stop_lookup, line_lookup = self._pas_station_dict()
#         self._stop_lookup = stop_lookup
#         self._line_lookup = line_lookup
#         self._stop_zone_map = TakstZones().stop_zone_map()
#         self._group_lines = _grouped_lines_dict(self._settings['config'])

#         self._alternates = set(chain(*ALTERNATE_STATIONS.values()))

#     def _pas_station_dict(self) -> Dict[Tuple[str, ...], List[int]]:
#         """
#         create the station dictionary from the underlying
#         dataset
#         """
#         lines = self._settings['lines']
#         pas_stations = _load_default_passenger_stations()

#         line_lookup = defaultdict(set)
#         for line in lines:
#             for frame in pas_stations:
#                 s = frame[line]
#                 good = set(s[s > 0].index)
#                 line_lookup[line].update(good)

#         stop_lookup = defaultdict(set)
#         for k, v in line_lookup.items():
#             for stopid in v:
#                 stop_lookup[stopid].add(k)

#         return stop_lookup, line_lookup

#     def get_ops(
#         self, stop_number: int,
#         format: Optional[str] = 'operator_id'
#         ) -> Tuple[Union[int, str], ...]:
#         """
#         Returns a tuple of the operators at the given station id

#         :param stop_number: uic number of the station
#         :type stop_number: int
#         :param format: 'operator_id', 'operator' or 'line', defaults to 'operator_id'
#             'operator' - returns str values representing the operators at the stop
#             'line' - returns line names of the stop

#         :type format: Optional[str], optional
#         :raises ValueError: if incorrect format is given
#         :return: the operators or lines serving the given station
#         :rtype: Tuple[Union[int, str], ...]

#         :Example:

#         to return the operators at Copenhagen central station:
#             the uic number is 8600626

#         >>> op_getter = StationOperators()
#         >>> cph_operator_ids = op_getter.get_ops(8600626, format='operator_id')
#         >>> cph_operator_ids
#         >>> (4, 8, 5, 6)
#         >>> cph_operators = op_getter.get_ops(8600626, format='operator')
#         >>> cph_operators
#         >>> ('first', 's-tog', 'dsb', 'metro')

#         Copenhagen central also has an s-tog platform that has
#         the uic number 8690626

#         in this case:

#         >>> cph_stog_operators = op_getter.get_ops(8690626, format='operator')
#         >>> cph_stog_operators
#         >>> ('s-tog', 'first', 'dsb', 'metro')

#         """

#         if format not in {'operator', 'operator_id', 'line'}:
#             raise ValueError(
#         "format must be one of 'operator', 'operator_id', 'line'"
#         )

#         fdict: Dict[str,  Union[Tuple[int, ...], Tuple[str, ...]]]
#         fdict = {
#             'operator_id': self._get_operator_id(stop_number),
#             'operator': self._get_operator(stop_number),
#             'line': self._get_line(stop_number)
#             }
#         return fdict[format]


#     def _check_bus_location(self, bus_stop_id: int) -> int:
#         """map a bus id to a station if possible, else 0

#         :param bus_stop_id: the stop number
#         :type bus_stop_id: int
#         :return: the mapped station number or 0 if no mapping exists
#         :rtype: int
#         """

#         return self.BUS_ID_MAP.get(bus_stop_id, 0)

#     def station_pair(
#         self,
#         start_uic: int,
#         end_uic: int,
#         format: Optional[str] = 'operator'
#         ) -> Union[Tuple[int, ...], Tuple[str, ...]]:
#         """
#         Returns the possible operators that can perform
#         the journey between a given station pair

#         :param start_uic: the uic number of the start station
#         :type start_uic: int
#         :param end_uic: the uic number of the end station
#         :type end_uic: int
#         :param format:  the way in which you desire the output, defaults to 'operator_id'
#             'operator_id' - returns values as integer ids representing operators
#             'operator' - returns values as strings of the operator name

#         :type format: Optional[str], optional
#         :raises ValueError: if the end stopid is not mappable to a station
#         :return: the operators serving the given station pair
#         :rtype: Tuple[int, ...]

#         :Example:

#         to return the possbible operators servicing a leg from copenhagen central
#         station to helsingør station

#         >>> op_getter = StationOperators(
#                 'kystkastrup', 'suburban',
#                 'sjællandfjernregional',
#                 'sjællandlocal', 'metro'
#                 )
#         >>> op_getter.station_pair(8600626, 8600669, format='line')
#             ['kystbanen']
#         >>> opgetter.station_pair(8600626, 8600669, format='operator')
#             ['first']
#         """
#         if format == 'operator':
#             try:
#                 return self.OP_CACHE[(start_uic, end_uic)]
#             except KeyError:
#                 pass
#         elif format == 'line':
#             try:
#                 return self.LINE_CACHE[(start_uic, end_uic)]
#             except KeyError:
#                 pass
#         else:
#             raise ValueError(f"format={format} not available")

#         start_bus = start_uic > MAX_RAIL_UIC or start_uic < MIN_RAIL_UIC
#         end_bus = end_uic > MAX_RAIL_UIC or end_uic < MIN_RAIL_UIC

#         if start_bus:
#             startzone = self._stop_zone_map.get(start_uic)
#             if not startzone:
#                 raise ValueError(f"Unknown bus stop id={start_uic}")
#             region = determine_takst_region(startzone)
#             line = {f'{region}_bus'}
#             operator = {self._settings['config']['bus'][region]}

#             self.OP_CACHE[(start_uic, end_uic)] = list(operator)
#             self.LINE_CACHE[(start_uic, end_uic)] = list(line)
#             if format == 'operator':
#                 return self.OP_CACHE[(start_uic, end_uic)]
#             return self.LINE_CACHE[(start_uic, end_uic)]


#         if not start_bus:
#             start_lines = self._stop_lookup[start_uic]
#         if not end_bus:
#             end_lines = self._stop_lookup[end_uic]
#         else:
#             bus_loc_check = self._check_bus_location(end_uic)
#             if not bus_loc_check:
#                  raise ValueError(f"Unknown bus stop id={end_uic}")
#             end_lines = self._stop_lookup[bus_loc_check]

#         line_intersection = start_lines.intersection(end_lines)

#         if not line_intersection:
#             # find missing stop point
#             start_groups = {self._group_lines.get(x) for x in start_lines}
#             start_groups = {x for x in start_groups if x}
#             end_groups = {self._group_lines.get(x) for x in end_lines}
#             end_groups = {x for x in end_groups if x}
#             group_intersection = start_groups.intersection(end_groups)

#             if group_intersection:
#                 group_lines, possible_operators = self._has_grp_intersection(
#                     group_intersection
#                     )
#                 self.OP_CACHE[(start_uic, end_uic)]  = possible_operators
#                 self.LINE_CACHE[(start_uic, end_uic)] = group_lines
#             else:
#                 self.OP_CACHE[(start_uic, end_uic)]  = []
#                 self.LINE_CACHE[(start_uic, end_uic)] = []
#         else:
#             possible_operators, possible_lines = self._has_line_intersection(
#                 start_uic, line_intersection
#                 )

#             self.OP_CACHE[(start_uic, end_uic)] = possible_operators
#             self.LINE_CACHE[(start_uic, end_uic)] = possible_lines

#         if format == 'operator':
#             return self.OP_CACHE[(start_uic, end_uic)]
#         return self.LINE_CACHE[(start_uic, end_uic)]

#     def _has_grp_intersection(self, group_intersection):

#         if len(group_intersection) == 1:
#             group = group_intersection.pop()
#             group_lines = set(self._settings['config'][group])
#             possible_operators = {self._settings['config'][x] for x in group_lines}

#         else:
#             group_lines = set()
#             possible_operators = set()

#         return list(group_lines), list(possible_operators)


#     def _has_line_intersection(self, start_uic, line_intersection):


#         if len(line_intersection) == 1:
#             line = list(line_intersection)
#             possible_operators = [self._settings['config'][line[0]]]
#             return possible_operators, line

#         suburban_option = any(
#             x in self._settings['config']['suburban'] for
#             x in line_intersection
#             )

#         metro_option = any(
#             x in self._settings['config']['metro'] for
#             x in line_intersection
#             )

#         if start_uic in mappers['s_uic'] and suburban_option:
#             possible_lines = {x for x in line_intersection if
#                               x in self._settings['config']['suburban']}
#             possible_operators = {
#                 self._settings['config'][x] for x in possible_lines
#                 }

#         elif start_uic in mappers['m_uic'] and metro_option:
#             possible_lines = {
#                 x for x in line_intersection if
#                 x in self._settings['config']['metro']
#                 }
#             possible_operators = {
#                 self._settings['config'][x] for x in possible_lines
#                 }

#         elif start_uic in self._alternates:
#             possible_lines = {
#                 x for x in line_intersection if
#                 x in self._settings['config']['sjællandlocal']
#                 }
#             possible_operators = {
#                 self._settings['config'][x] for x in possible_lines
#                 }
#         else:
#             possible_lines = {
#                 x for x in line_intersection if
#                 x not in self._settings['config']['suburban'] and
#                 x not in self._settings['config']['metro'] and
#                 x not in self._settings['config']['sjællandlocal']
#                 }
#             possible_operators = {
#                 self._settings['config'][x] for x in possible_lines
#                 }
#         return list(possible_operators), list(possible_lines)
# """