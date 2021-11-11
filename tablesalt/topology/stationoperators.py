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
from tablesalt.transitfeed.feed import TransitFeed, BusMapper

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

    def __init__(
        self,
        feed: TransitFeed,
        bus_distance_cutoff: int = 500,
        allow_operator_legs: bool = False,
        crs: int = 25832
        ) -> None:
        """[summary]

        :param feed: [description]
        :type feed: TransitFeed
        :param bus_distance_cutoff: [description], defaults to 500
        :type bus_distance_cutoff: int, optional
        :param crs: [description], defaults to 25832
        :type crs: int, optional
        """

        self.feed = feed

        self._suburban_operator, self._metro_operator, self._local_operator = \
            self._determine_operators()

        self.bus_mapper = BusMapper(self.feed, crs=crs)
        self.bus_to_station_map: Dict[int, int] = \
            self.bus_mapper.get_bus_map(bus_distance_cutoff=bus_distance_cutoff)

        self.station_to_bus_map = self._reverse_bus_map()
        self._lookup = self._process_stop_times()

        self._allow_operator_legs = allow_operator_legs

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
        # this must be changed in future for each local line

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
        # kast_agency_name = self.feed.get_agency_name_for_trip(kastrup_trip)
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

        metro = {k: v for k, v in relation_operators.items() if k[1] in METRO_UIC and k[0] not in METRO_UIC}
        suburban = {k: v for k, v in relation_operators.items() if k[1] in SUBURBAN_UIC and k[0] not in SUBURBAN_UIC}
        buses = {k: v for k, v in relation_operators.items() if k[1] in self.station_to_bus_map}
        buses_start = {k: v for k, v in relation_operators.items() if k[1] in self.bus_to_station_map}

        st_to_bus = {}
        for k, v in buses.items():
            new_legs = tuple((k[0], x) for x in self.station_to_bus_map[k[1]])
            new = {x: v for x in new_legs}
            st_to_bus.update(new)


        bus_to_st = {}
        for k, v in buses_start.items():
            leg = (k[0], self.bus_to_station_map[k[1]])
            mapped_leg = (leg[0], SUBURBAN_UIC.get(leg[1], leg[1]))
            mapped_leg_l = (leg[0], LOCAL_UIC.get(leg[1], leg[1]))
            mapped_leg_m = (leg[0], METRO_UIC.get(leg[1], leg[1]))
            if leg not in relation_operators:
                bus_to_st[leg] = v
            if mapped_leg not in relation_operators:
                bus_to_st[mapped_leg] = v
            if mapped_leg_l not in relation_operators:
                bus_to_st[mapped_leg_l] = v
            if mapped_leg_m not in relation_operators:
                bus_to_st[mapped_leg_m] = v

        new_metro = {(k[0], METRO_UIC[k[1]]): v for k, v in metro.items()}
        new_sub = {(k[0], SUBURBAN_UIC[k[1]]): v for k, v in suburban.items()}

        return {
            # **st_to_bus,
            **bus_to_st,
            **new_metro,
            **new_sub
            }

    def _start_stop_changes(self, relation_operators):

        suburban = {k: v for k, v in relation_operators.items() if k[0] in SUBURBAN_UIC}
        new_suburban = {
            (SUBURBAN_UIC[k[0]], k[1]): v for k, v in suburban.items()
        }

        local = {k: v for k, v in relation_operators.items() if k[0] in LOCAL_UIC}
        new_local = {
            (LOCAL_UIC[k[0]], k[1]): v for k, v in local.items()
        }

        metro = {k: v for k, v in relation_operators.items() if k[0] in METRO_UIC}
        new_metro = {
            (METRO_UIC[k[0]], k[1]): v for k, v in metro.items()
        }


        return {**new_suburban, **new_local, **new_metro}

    def _process_stop_times(self) -> Dict[int, Tuple[Tuple[int, int]]]:
        """
        This method creates the dictionary in self._lookup
        It uses the stop_times dataset from the given transit feed
        """
        relation_operators = defaultdict(set)

        seen = set()
        for trip_id, stoptime in self.feed.stop_times.data.items():
            stopids = tuple(x['stop_id'] for x in stoptime)
            if stopids in seen:
                continue
            perms = set(permutations(stopids, 2))
            seen.add(stopids)
            agency = self.feed.get_agency_name_for_trip(trip_id)
            for stop_relation in perms:
                relation_operators[stop_relation].add(agency)
            # this deals only with sjælland correctly
            # needs more work for jylland maybe
            if agency == self._suburban_operator:
                updated_leg_permutations = self._suburban_stop_changes(perms)
            elif agency == self._local_operator:
                updated_leg_permutations = self._local_stop_changes(perms)
            elif agency == self._metro_operator:
                updated_leg_permutations = self._metro_stop_changes(perms)
            else:
                updated_leg_permutations = {}

            relation_operators.update(updated_leg_permutations)

        relation_operators = dict(relation_operators)
        #deal with the alternate rejsekort station numbers
        alts = self._alternate_stop_numbers(relation_operators)
        relation_operators.update(alts)

        end_changes = self._end_stop_changes(relation_operators)
        relation_operators.update(end_changes)

        start_changes = self._start_stop_changes(relation_operators)
        relation_operators.update(start_changes)
        relation_operators = {k: tuple(v) for k, v in relation_operators.items()}

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

    def _same_operator_leg(self, start_stop_id, end_stop_id):

        start_transfer_options = {
            k[1] for k, v in self._lookup.items() if k[0] == start_stop_id
            }
        end_transfer_options = {
            k[0]  for k, v in self._lookup.items() if k[1] == end_stop_id
            }

        transfer_options = start_transfer_options.intersection(end_transfer_options)

        for transfer in transfer_options:
            start_op = set(self._lookup[(start_stop_id, transfer)])
            end_op = set(self._lookup[(transfer, end_stop_id)])
            inter = start_op.intersection(end_op)
            if inter:
                return tuple(inter)
        raise KeyError

    def station_pair(
        self,
        start_stop_id: int,
        end_stop_id: int,
        ) -> Tuple[str]:
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

        msg = (f"No operator found servicing leg {start_stop_id}, {end_stop_id} "
               f"in the period {self.feed.feed_period()}")
        try:
            return self._lookup[(start_stop_id, end_stop_id)]
        except KeyError:
            if self._allow_operator_legs:
                try:
                    operator = self._same_operator_leg(start_stop_id, end_stop_id)
                    self._lookup[(start_stop_id, end_stop_id)] = operator
                    return operator
                except KeyError:
                    raise KeyError(msg)
            raise KeyError(msg)
