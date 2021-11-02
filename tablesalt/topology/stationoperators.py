# -*- coding: utf-8 -*-
"""
Classes to interact with passenger stations and the operators that serve them
"""
#standard imports
import ast
import json
from collections import defaultdict
from itertools import groupby, permutations, product
from operator import itemgetter
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
REV_SUBURBAN_UIC = {v:k for k, v in SUBURBAN_UIC.items()}

METRO_UIC = dict(CONFIG['metro_platform_numbers'])
METRO_UIC = {int(k): int(v) for k, v in METRO_UIC.items()}
REV_METRO_UIC = {v:k for k, v in METRO_UIC.items()}

METRO_SUBURBAN = {k: SUBURBAN_UIC.get(v, v) for k, v in REV_METRO_UIC.items()}

LOCAL_UIC_1 =  dict(CONFIG['local_platform_numbers_1'])
LOCAL_UIC_1 = {int(k): int(v) for k, v in LOCAL_UIC_1.items()}

LOCAL_UIC_2 =  dict(CONFIG['local_platform_numbers_2'])
LOCAL_UIC_2 = {int(k): int(v) for k, v in LOCAL_UIC_2.items()}

LOCAL_UIC = {**LOCAL_UIC_1, **LOCAL_UIC_2}
REV_LOCAL_UIC = {v:k for k, v in METRO_UIC.items()}

ALTERNATE_UIC =  dict(CONFIG['alternate_numbers'])
ALTERNATE_UIC = {int(k): ast.literal_eval(v) for k, v in ALTERNATE_UIC.items()}


SUBURBAN_TEST_STOPS = set(ast.literal_eval(CONFIG['suburban_farum_cph']['test_stops']))
METRO_TEST_STOPS = set(ast.literal_eval(CONFIG['metro_lindevang']['test_stops']))
KASTRUP_TEST_STOPS =  set(ast.literal_eval(CONFIG['kastrup_cph']['test_stops']))
LOCAL_TEST_STOPS =  set(ast.literal_eval(CONFIG['local']['test_stops']))


M_RANGE: Set[int] = set(range(8603301, 8603400))
S_RANGE: Set[int] = set(range(8690000, 8699999))

# this is temporary, must change this. Use Transit feed to see if the stops service buses.
MAX_RAIL_UIC: int = 9999999
MIN_RAIL_UIC: int = 7400000

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


class StationOperators:

    def __init__(self, feed: TransitFeed) -> None:
        """class to determine the operator between stop point ids

        :param feed: a gtfs transitfeed from rejseplanen
            The feed can be created from using tablesalt.transitfeed.archived_transitfeed,
            or the very latest available feed using tablesalt.transitfeed.latest_transitfeed,

        :type feed: TransitFeed
        """

        self.feed = feed
        self._suburban_operator, self._metro_operator, self._local_operator = \
            self._determine_operators()

        self.bus_to_station_map: Dict[int, int] = _load_bus_station_map()
        self.station_to_bus_map = self._reverse_bus_map()
        self._lookup = self._process_stop_times()

    def _reverse_bus_map(self):
        bmap = tuple(self.bus_to_station_map.items())
        bmap = sorted(bmap, key=itemgetter(1))

        station_to_bus = {
            key: tuple(x[0] for x in grp) for
            key, grp in groupby(bmap, key=itemgetter(1))
            }

        suburban = {SUBURBAN_UIC[k]: v for k, v in station_to_bus.items() if k in SUBURBAN_UIC}
        local = {LOCAL_UIC[k]: v for k, v in station_to_bus.items() if k in LOCAL_UIC}

        return {**station_to_bus, **suburban, **local}

    def _determine_operators(self):

        stoptimes = list(self.feed.stop_times.data.items())

        suburban_trip = None
        metro_trip = None
        kastrup_trip = None
        local_trip = None  # NOTE. only one local operator for sjælland (must fix this)

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

    @staticmethod
    def _start_operator_changes(stop_map, leg_permutations, operator):
        start_perms = {(stop_map.get(x[0], x[0]), x[1]) for x in leg_permutations}
        new_perms = start_perms - leg_permutations
        if new_perms:
            return {stop_relation: {operator} for stop_relation in new_perms}

        return {}

    @staticmethod
    def _end_operator_changes(stop_map, leg_permutations, operator):
        end_perms = {(x[0], stop_map.get(x[1], x[1])) for x in leg_permutations}
        new_perms = end_perms - leg_permutations
        if new_perms:
            return {stop_relation: {operator} for stop_relation in new_perms}
        return {}

    def _suburban_stop_changes(self, leg_permutations):
        # legs starting at 869 platform must only be suburban operator
        s_suburban_ops = self._start_operator_changes(
            SUBURBAN_UIC, leg_permutations, self._suburban_operator
            )
        e_suburban_ops = self._end_operator_changes(
            SUBURBAN_UIC, leg_permutations, self._suburban_operator
            )
         # if starting platform is local num 861/862... the operator should still be suburban
        s_local_ops = self._start_operator_changes(
            LOCAL_UIC, leg_permutations, self._suburban_operator
            )
        # if end platform is local num 861/862... the operator should still be suburban
        e_local_ops = self._end_operator_changes(
                LOCAL_UIC, leg_permutations, self._suburban_operator
            )
        # if end number is a metro number operator still suburban
        e_metro_ops = self._end_operator_changes(
                METRO_UIC, leg_permutations, self._suburban_operator
            )
        return {
            **s_suburban_ops,
            **e_suburban_ops,
            **s_local_ops,
            **e_local_ops,
            **e_metro_ops
            }

    def _local_stop_changes(self, leg_permutations):
        s_local_ops = self._start_operator_changes(
            LOCAL_UIC, leg_permutations, self._local_operator
            )

        e_local_ops = self._end_operator_changes(
                LOCAL_UIC, leg_permutations, self._local_operator
            )

        s_suburban_ops = self._start_operator_changes(
            SUBURBAN_UIC, leg_permutations, self._local_operator
            )

        return  {**s_suburban_ops, **s_local_ops, **e_local_ops}

    def _metro_stop_changes(self, leg_permutations):

        metro_ops = self._end_operator_changes(
            REV_METRO_UIC, leg_permutations, self._metro_operator
            )
        suburban_ops = self._end_operator_changes(
            METRO_SUBURBAN, leg_permutations, self._metro_operator
            )

        return {**metro_ops, **suburban_ops}

    def _end_stop_changes(self, relation_operators):

        metro = {k: v for k, v in relation_operators.items() if k[1] in METRO_UIC}
        suburban = {k: v for k, v in relation_operators.items() if k[1] in SUBURBAN_UIC}
        buses = {k: v for k, v in relation_operators.items() if k[1] in self.station_to_bus_map}
        buses_start = {k:v for k, v in relation_operators.items() if k[1] in self.bus_to_station_map}

        st_to_bus = {}
        for k, v in buses.items():
            new_legs = ((k[0], x) for x in self.station_to_bus_map[k[1]])
            new = {x: v for x in new_legs}
            st_to_bus.update(new)

        bus_to_st = {}
        for k, v in buses_start.items():
            leg = (k[0], self.bus_to_station_map[k[1]])
            if leg not in relation_operators:
                bus_to_st[leg] = v

        new_metro = {(k[0], METRO_UIC[k[1]]): v for k, v in metro.items()}
        new_sub = {(k[0], SUBURBAN_UIC[k[1]]): v for k, v in suburban.items()}

        return {
            **st_to_bus,
            **bus_to_st,
            **new_metro,
            **new_sub
            }

    def _process_stop_times(self) -> Dict[int, Tuple[Tuple[int, int]]]:
        """
        This method creates the dictionary in self._lookup
        It uses the stop_times dataset from the given transit feed
        """
        relation_operators = {}

        seen = set()
        for trip_id, stoptime in self.feed.stop_times.data.items():
            stopids = (x['stop_id'] for x in stoptime)
            perms = set(permutations(stopids, 2))
            if tuple(perms) in seen: # perms must remain a set
                continue
            seen.add(tuple(perms))
            agency = self.feed.get_agency_name_for_trip(trip_id)
            for stop_relation in perms:
                try:
                    relation_operators[stop_relation].add(agency)
                except KeyError:
                    relation_operators[stop_relation] = {agency}
            # this deals only with sjælland correctly
            # needs more work for jylland
            if agency == self._suburban_operator:
                updated_leg_permutations = self._suburban_stop_changes(perms)
            elif agency == self._local_operator:
                updated_leg_permutations = self._local_stop_changes(perms)
            elif agency == self._metro_operator:
                updated_leg_permutations = self._metro_stop_changes(perms)
            else:
                updated_leg_permutations = {}

            relation_operators.update(updated_leg_permutations)
        #deal with the alternate rejsekort station numbers
        alts = self._alternate_stop_numbers(relation_operators)
        relation_operators.update(alts)

        end_changes = self._end_stop_changes(relation_operators)
        relation_operators.update(end_changes)

        return relation_operators

    def _alternate_stop_numbers(self, relation_operators):
        """"replace the station numbers with their alternate numbers
        so that the alternate stations are also in the lookup"""
        alts = {}
        for k, v in relation_operators.items():
            if any(x in ALTERNATE_UIC for x in k):
                start = ALTERNATE_UIC.get(k[0], k[0])
                end = ALTERNATE_UIC.get(k[1], k[1])
                if isinstance(start, int):
                    start = [start]
                if isinstance(end, int):
                    end = [end]
                new_keys = product(start, end)
                for nk in new_keys:
                    alts[nk] = v
        return alts

    def station_pair(
        self,
        start_stop_id: int,
        end_stop_id: int
        ) -> Set[str]:
        """Determine the possible operators for a pair of stations.
        Can be used to determine the operator servicing a leg of a trip

        If no operator services the given station pair, a KeyError is raised
        :param start_stop_id: the station number of the starting station
        :type start_stop_id: int
        :param end_stop_id: the station number of the end station
        :type end_stop_id: int
        :return: a set of possible operators servicing the station pair
        :rtype: Set[str]
        """

        # if start bus-end train, check end station and stops around it...must be a bus leg to one of the stops

        # if start train - end bus, find the closest station to the bus stop and check leg
        try:
            return self._lookup[(start_stop_id, end_stop_id)]
        except KeyError:
            raise KeyError(
                f"No operator found servicing leg {start_stop_id}, {end_stop_id}"
                )


# make a separate lookup class
# class StationOperators():

#     BUS_ID_MAP: Dict[int, int] = _load_bus_station_map() self.station_pair(8600669, 6584)

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