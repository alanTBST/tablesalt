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
from typing import Any, AnyStr, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import pandas as pd  # type: ignore
import pkg_resources
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
        self._transfer_cache: DefaultDict[int, Set[int]] = defaultdict(set)
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
        local_trip = None  # TODO NOTE. only one local operator for sjælland (must fix this)

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

    def _chain_bus_permutations(self, perms: Tuple[Tuple[int, int]]):

        output = []
        for start_id, end_id in perms:
            end_bus_ids = self.station_to_bus_map.get(end_id, end_id)
            if end_id != end_bus_ids:
                new = [(start_id, x) for x in end_bus_ids]
                output.extend(new)

        return set(output)

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
            bus_perms = self._chain_bus_permutations(perms)
            all_perms = perms | bus_perms
            all_perms = sorted(all_perms, key=lambda x: x[0])
            tfers = {k: set(x[1] for x in v) for k, v in groupby(all_perms, lambda x: x[0])}
            for k, v in tfers.items():
                self._transfer_cache[k].update(v)
            seen.add(stopids)
            agency = self.feed.get_agency_name_for_trip(trip_id)
            try:
                agency = agency.lower()
            except AttributeError:
                pass
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

        start_transfer_options = self._transfer_cache.get(start_stop_id, set())
        end_transfer_options = self._transfer_cache.get(end_stop_id, set())

        transfer_options = start_transfer_options.intersection(end_transfer_options)

        if not transfer_options:
            raise KeyError
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
