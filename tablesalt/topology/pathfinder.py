# -*- coding: utf-8 -*-
"""
Functions and Classes used to find
all the zones travelled through on
a trip. Calculated from the given zones
in which a user taps in or out

"""

from itertools import chain, groupby
from collections import Counter
from functools import lru_cache
from typing import Tuple, Union, Optional, Dict, Any
import pkg_resources

import pandas as pd  #type: ignore
import numpy as np  #type: ignore
from networkx.classes.graph import Graph #type: ignore

from tablesalt.common import triptools
from tablesalt.common.io import mappers
from tablesalt.topology import stationoperators
from tablesalt.topology.stopnetwork import StopsList
from tablesalt.topology.zonegraph import ZoneGraph
from tablesalt.topology.tools import determine_takst_region

# put these in lines in a config
OPGETTER = stationoperators.StationOperators(
    'kystbanen', 'local', 'metro', 'suburban', 'fjernregional'
    )

OP_MAP = {v: k.lower() for k, v in mappers['operator_id'].items()}
REV_OP_MAP = {v: k for k, v in OP_MAP.items()}
rev_model_dict = {v:k for k, v in mappers['model_dict'].items()}
CO_TR = (rev_model_dict['Co'], rev_model_dict['Tr'])


#TODO load from config with year
SOLO_ZONE_PRIS = {
    'movia_h': {
        'dsb': 6.38,
        'movia_h': 9.18,
        'first': 6.42,
        'stog': 7.12,
        's-tog': 7.12,
        'metro': 9.44
        },
    'movia_s': {
        'dsb': 6.55,
        'movia_s': 7.94,
        },
    'movia_v': {
        'dsb': 6.74,
        'movia_v': 8.43,
        },
    'dsb': {
        'dsb': 6.57,
        'movia_h': 6.36,
        'movia_s': 6.36,
        'movia_v': 6.36,
        }
    }

def load_border_stations() -> Dict[int, Tuple[int, ...]]: # put this in TBSTtopology
    "load the border stations dataset from package resources"
    fp = pkg_resources.resource_filename(
        'tablesalt',
        'resources/revenue/borderstations.xlsx'
        )
    border_frame = pd.read_excel(fp)
    border_frame['Zones'] = list(zip(
        border_frame['Zone1'], border_frame['Zone2'], border_frame['Zone3']
        ))

    border_dict = {
        k: tuple(int(x) for x in v if x > 0)
        for k, v in border_frame.set_index('UIC')['Zones'].to_dict().items()
        }

    s_dsb = [x - 90000 for x in mappers['s_uic']]
    inborder = set(s_dsb).intersection(border_dict)
    inborder_added = {k + 90000: border_dict[k] for k in inborder}

    return {**border_dict, **inborder_added}

BORDER_STATIONS = load_border_stations()

@lru_cache(2**16)
def _is_bus(stopid: int) -> bool:

    return (stopid > stationoperators.MAX_RAIL_UIC or
            stopid < stationoperators.MIN_RAIL_UIC)

@lru_cache(2**16)
def impute_leg(g: Graph, zone_leg: Tuple[int, int]) -> Tuple[int, ...]:
    """fill in the total zone path of a zone leg

    :param g: a networkx graph
    :type g: Graph
    :param zone_leg: a tuple of the zone leg eg (1001, 1004)
    :type zone_leg: Tuple[int, int]
    :return: a filled in zone path
    :rtype: Tuple[int, ...]
    """

    return g.shortest_path(*zone_leg)[0]

@lru_cache(2**16)
def impute_zone_legs(
    g: Graph,
    trip_zone_legs: Tuple[Tuple[int, int], ...]
    ) ->  Tuple[Tuple[int, ...], ...]:
    """fill in the paths of the trip zone legs

    :param g: a networkx graph
    :type g: Graph
    :param trip_zone_legs: tuple of zone leg tuples eg ((1001, 1004), (1004,1001))
    :type trip_zone_legs: Tuple[Tuple[int, int], ...]
    :return: a tuple of filled in zone paths
    :rtype: Tuple[Tuple[int, ...], ...]
    """
    return tuple(impute_leg(g, leg)
                 for leg in trip_zone_legs)

@lru_cache(2**16)
def get_touched_zones(zone_sequence: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    get the zones in which the card has been checked,
    preserving the order of taps
    """
    touched = []
    seen = set()
    for i, j in enumerate(zone_sequence):
        if i == 0:
            touched.append(j)
            seen.add(j)
            continue
        if not j in seen or not j == zone_sequence[i-1]:
            touched.append(j)
            seen.add(j)
    return tuple(touched)

@lru_cache(2**16)
def to_legs(sequence: Tuple[int, ...]) ->Tuple[int, ...]:
    """convert a tuple to leg form

    :param sequence: a tuple of zones, stops, etc
    :type sequence: Tuple[int, ...]
    :return: the legified form
    :rtype: Tuple[int, ...]
    """
    return triptools.sep_legs(sequence)

@lru_cache(2**16)
def _to_legs_stops(stop_legs):
    """convert to legs and map the bus id to station if possible"""
    return tuple(
            (leg[0], OPGETTER.BUS_ID_MAP.get(leg[1])) if
            (not _is_bus(leg[0]) and _is_bus(leg[1])) else leg
            for leg in stop_legs
            )

@lru_cache(2**16)
def _chain_vals(vals):
    """chain zones together for legs removing duplicates at end"""
    lvals = len(vals)
    visited_zones = tuple(
        chain(*[j[:-1] if i != lvals - 1 else j
              for i, j in enumerate(vals)])
        )
    return visited_zones


class ZoneProperties():
    "ZoneProperties"

    VISITED_CACHE = {}

    def __init__(self,
                 graph: ZoneGraph,
                 zone_sequence: Tuple[int, ...],
                 stop_sequence: Tuple[int, ...],
                 region: Optional[str] = 'sjælland'
                 ) -> None:
        """
        Class to determine how many zones are travelled through,
        whether a border zone is touched, how the border affects the
        total number or zones, etc

        :param graph: an instance of the zonegraph class
        :type graph: ZoneGraph
        :param zone_sequence: a tuple of the zone sequence of a trip
        :type zone_sequence: Tuple[int, ...]
        :param stop_sequence:  a tuple of the stop sequence of a trip
        :type stop_sequence: Tuple[int, ...]
        :param region: the region to use, defaults to 'sjælland'
        :type region: Optional[str], optional
        :return: ''
        :rtype: None

        """

        self.graph = graph
        # self.ring_dict = self.graph.ring_dict(region)
        self.zone_sequence: Tuple[int, ...] = zone_sequence
        self.stop_sequence: Tuple[int, ...] = stop_sequence

        self.stop_legs: Tuple[Tuple[int, ...], ...] = to_legs(stop_sequence)
        self.stop_legs = _to_legs_stops(self.stop_legs)
        self.zone_legs: Tuple[Tuple[int, ...], ...] = to_legs(zone_sequence)

        self.border_trip: bool = False
        self.border_legs: Union[Tuple[()], Tuple[int, ...]]  = ()

        if any(x in BORDER_STATIONS for x in chain(*self.stop_legs)):
            self.border_trip = True

        self.border_legs = self._border_touch_legs() if \
            self.border_trip else self.border_legs

        self.touched_zones = get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = to_legs(self.touched_zones)

    def _border_touch_legs(self) -> Tuple[int, ...]:
        """return the card touch positions on the trip
        as a tuple"""

        return tuple(i for i, j in enumerate(self.stop_legs)
                     if any(x in BORDER_STATIONS for x in j))


    def _visited_zones_on_leg(self, zone_leg: Tuple[int, int]) -> None:
        """for a tuple of start and end zone
        find the zones travelled through
        and update the cache"""

        if len(set(zone_leg)) == 1:
            visited = (zone_leg[0],) # note tuple
        else:
            all_short_paths = self.graph.shortest_path(
                zone_leg[0], zone_leg[1]
                )
            if len(all_short_paths) > 1:
                options = np.array([x for x in all_short_paths]).T
                visited = (tuple(set(x)) for x in options)
                visited = tuple(int(x[0]) if len(x) == 1 else x for x in visited)
            else:
                visited = all_short_paths[0]
        self.VISITED_CACHE[zone_leg] = visited

    def get_visited_zones(
            self,
            zone_legs: Tuple[Tuple[int, int], ...]
            ) -> Tuple[Union[int, Tuple[int, ...]], ...]:
        """visited zones is a list of the zones visited on a trip,
        in order, but removing adjacent duplicate zones"""

        vals = []
        for leg in zone_legs:
            try:
                vals.append(self.VISITED_CACHE[leg])
            except KeyError:
                self._visited_zones_on_leg(leg)
                vals.append(self.VISITED_CACHE[leg])

        visited_zones = _chain_vals(tuple(vals))


        if not visited_zones:
            return tuple(self.touched_zones)

        return tuple(visited_zones)

    def _chain_zones(self, visited_zones):
        # chain visited together preserving order
        chained_zones = []
        try:
            for x in visited_zones:
                if isinstance(x, int):
                    chained_zones.append(x)
                else:
                    for y in x:
                        chained_zones.append(y)
        except TypeError:
            chained_zones = list(visited_zones)
        return chained_zones

    def _borderless_properties(self) -> Dict[str, Any]:
        """get the properties if the trip does not touch a border station

        :return: a dictionary of properties
        :rtype: Dict[str, Any]
        """

        visited_zones = self.get_visited_zones(self.touched_zone_legs)

        chained_zones = self._chain_zones(visited_zones)

        double_back = bool(max(Counter(chained_zones).values()) > 1)
        imputed_zone_legs = impute_zone_legs(self.graph, self.zone_legs)

        prop_dict = {
            'visited_zones': visited_zones,
            'total_travelled_zones': len(visited_zones),
            'double_back': double_back,
            'touched_zones': self.touched_zones,
            'stop_sequence': self.stop_sequence,
            'stop_legs': self.stop_legs,
            'zone_sequence': self.zone_sequence,
            'zone_legs': self.zone_legs,
            'border_legs': self.border_legs,
            'imputed_zone_legs': imputed_zone_legs,
            'nlegs_in_touched': {
                tzone: len([x for x in self.zone_legs if tzone in x])
                       for tzone in self.touched_zones
                },
            'zone_legs_regions': tuple(determine_takst_region(x) for x in imputed_zone_legs)
            }

        return prop_dict


    def _bordered_properties(self):
        """get the properties of the trip if it touches a border station
        """

        zone_legs = self._zone_legs_for_border_stations()

        # make properties setter
        self.zone_sequence = tuple(
            [x[0] for x in zone_legs] + [zone_legs[-1][1]]
            )
        
        self.zone_legs = to_legs(self.zone_sequence)
        self.touched_zones = get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = to_legs(self.touched_zones)
        #---------------

        return self._borderless_properties()

    def _zone_legs_for_border_stations(self):
        zone_legs = []
        for legnr, zone_leg in enumerate(self.zone_legs):
            start_zone = zone_leg[0]
            end_zone = zone_leg[1]
            if legnr in self.border_legs:
                stop_leg = self.stop_legs[legnr]
                if stop_leg[0] in BORDER_STATIONS:
                    start_zone = self._border_startzone(zone_leg, stop_leg)
                if stop_leg[1] in BORDER_STATIONS:
                    end_zone = self._border_endzone(zone_leg, stop_leg)

            leg = (start_zone, end_zone)
            zone_legs.append(leg)
        return zone_legs

    def _border_endzone(self, zone_leg, stop_leg):
        border_zones = BORDER_STATIONS[stop_leg[1]]
        border_distances = [len(self.graph.shortest_path(zone_leg[0], x)[0])
                            for x in border_zones]
        min_pos = int(np.argmin(border_distances))
        end_zone = border_zones[min_pos]
        return end_zone

    def _border_startzone(self, zone_leg, stop_leg):
        border_zones = BORDER_STATIONS[stop_leg[0]]
        border_distances = [len(self.graph.shortest_path(x, zone_leg[1])[0]) 
                            for x in border_zones]
        min_pos = int(np.argmin(border_distances))
        start_zone = border_zones[min_pos]
        return start_zone

    def property_dict(self) -> Dict[str, Any]:
        """
        Return the zone property dictionary for the trip

        :return: dictionary of trip properties
        :rtype: Dict[str, Any]

        """

        if not self.border_trip:
            return self._borderless_properties()

        return self._bordered_properties()

def _remove_idxs(idxs, legs):
    """remove an item from a sequence of legs"""
    return tuple(j for i, j in enumerate(legs) if i not in idxs)

def legops(new_legs):
    """just legify the output from determin_operator"""

    out = []
    for x in new_legs:
        if len(x) == 1:
            out.append((x[0], x[0]))
        else:
            out.append((x[0], x[1]))
    return tuple(out)

def operators_in_touched_(
        tzones: Tuple[int, ...],
        zonelegs: Tuple[Tuple[int, ...]],
        oplegs: Tuple[Tuple[int, ...]]
        ) -> Dict[int, Tuple[int, ...]]:
    """
    determine the operators in the zones that are touched by the user

    :param tzones: a tuple of the zones touched by a rejsekort tap on a trip
    :type tzones: Tuple[int, ...]
    :param zonelegs: a legified tuple of zones
    :type zonelegs: Tuple[Tuple[int, ...]]
    :param oplegs: a legified tuple of operators
    :type oplegs: Tuple[Tuple[int, ...]]
    :return: Tuple[Tuple[int, ...]]
    :rtype: Dict[int, Tuple[int, ...]]:

    """

    ops_in_touched = {}
    for tzone in tzones:
        ops = []
        for i, j in enumerate(zonelegs):
            if tzone in j:
                l_ops = list(oplegs[i])
                ops.extend(l_ops)
        # modulo:  only put in one operator value per leg
        ops_in_touched[tzone] = \
        tuple(j for i, j in enumerate(ops) if i % 2 == 0)

    return ops_in_touched

@lru_cache(2**16)
def aggregated_zone_operators(v):
    "aggregate all the operators for the zone shares"

    vals = list(v)

    out_list = []
    for x in vals:
        if isinstance(x[0], int):
            out_list.append(x)
            continue
        for y in x:
            out_list.append(y)
    out_list = [(x[0], OP_MAP[x[1]]) for x in out_list]
    # if any(isinstance(x[1], tuple) for x in out_list):
    #     out_list = [x if isinstance(x[1], int) else (x[0], x[1][0]) for x in out_list]

    out_list = sorted(out_list, key=lambda x: x[1])

    return tuple(((sum(x[0] for x in grp), key)) for
                 key, grp in groupby(out_list, key=lambda x: x[1]))

class ZoneSharer(ZoneProperties):

    SHARE_CACHE = {}

    def __init__(
            self,
            graph: ZoneGraph,
            zone_sequence: Tuple[int, ...],
            stop_sequence: Tuple[int, ...],
            operator_sequence: Tuple[int, ...],
            usage_sequence: Tuple[int, ...]
            ) -> None:
        """Main class to determine the zone shares for each operator
        on a trip.

        :param graph: the zonegraph that is created from the shapes.txt and
            takst zones polygons
        :type graph: ZoneGraph
        :param zone_sequence: a tuple of zone numbers from the delrejser data
        :type zone_sequence: Tuple[int, ...]
        :param stop_sequence: a tuple of stop ids from the delrejser data
        :type stop_sequence: Tuple[int, ...]
        :param operator_sequence: a tuple of operator ids corr. to the operator names
        :type operator_sequence: Tuple[int, ...]
        :param usage_sequence: a tuple of model ids
        :type usage_sequence: Tuple[int, ...]
        """

        super().__init__(graph, zone_sequence, stop_sequence)

        self.zone_sequence = zone_sequence
        self.stop_sequence = stop_sequence
        self.operator_sequence = operator_sequence
        self.usage_sequence = usage_sequence
        self.operator_legs = to_legs(self.operator_sequence)
        self.usage_legs = to_legs(self.usage_sequence)

        self.single: bool = self._is_single()

        self.region: str = determine_takst_region(self.zone_sequence)


    def _is_single(self) -> bool:
        """has only one operator"""
        return len(set(chain(*self.operator_legs))) == 1

    def _remove_cotr(self) -> None:
        """remove all of the legs that are cotr touches
        for each relevant attribute
        """

        if CO_TR not in self.usage_legs:
            return
        cotr_idxs = ()
        for i, j in enumerate(self.usage_legs):
            if j == CO_TR:
                cotr_idxs += (i,)

        self.stop_legs = _remove_idxs(cotr_idxs, self.stop_legs)
        self.operator_legs = _remove_idxs(cotr_idxs, self.operator_legs)
        self.zone_legs = _remove_idxs(cotr_idxs, self.zone_legs)
        self.usage_legs = _remove_idxs(cotr_idxs, self.usage_legs)

    def _station_operators(self):
        """get the operators at the visited stations"""

        oplegs = tuple(
            OPGETTER.station_pair(*x, format='operator_id')
            for x in self.stop_legs
                )

        return oplegs

    def _share_single(self, val):
        """assign the single operator the zones"""
        shares = (
            self.property_dict()['total_travelled_zones'],
            OP_MAP[self.operator_sequence[0]]
            )
        self.SHARE_CACHE[val] = shares
        return shares

    def share_calculation(
        self,
        properties: Dict[str, Any]
        ) -> Union[Tuple[int, str], Tuple[Tuple[int, str], ...]]:
        """
        Calculate the zone shares for the operators on the trip

        :param val: property_dict from ZoneProperties
        :type val: Dict[str, Any]
        :return: the zone shares for the trip
        :rtype: Union[Tuple[int, str], Tuple[Tuple[int, str], ...]]

        """
        # use a default dict here
        out = {}
        out_solo = {}

        for i, imputed_leg in enumerate(properties['imputed_zone_legs']):
            break
            region = properties['zone_legs_regions'][i]
            op_id = self.operator_legs[i][0]
            for zone in imputed_leg:
                if zone in properties['nlegs_in_touched']:
                    if properties['nlegs_in_touched'][zone] == 1:                      
                        out[zone] = (1, op_id)
                        out_solo[zone] = (1 * SOLO_ZONE_PRIS[region][OP_MAP[op_id]], op_id)
                    else:
                        break
                        # need to assign the fraction of the zone for the leg based on itøs region for solo    
                        counts = Counter(properties['ops_in_touched'][zone])
                        solo_counts = Counter(OP_MAP[x] for x in properties['ops_in_touched'][zone])
                        try:
                            out[zone] = tuple((v/properties['nlegs_in_touched'][zone], k) for
                                              k, v in counts.items())
                        except ZeroDivisionError:
                            out[zone] = 1, op_id
                            out_solo[zone] = (1 * SOLO_ZONE_PRIS[region][OP_MAP[op_id]], op_id)
                else:
                    out[zone] = 1, op_id
                    out_solo[zone] = (1 * SOLO_ZONE_PRIS[region][OP_MAP[op_id]], op_id)

        return aggregated_zone_operators(tuple(out.values()))

    @staticmethod
    def _standardise(share):
        """remove the takstsæt from movia operators
            eg 'movia_h' -> 'movia'
        """

        if isinstance(share[0], int):
            # removes _h, _s, _v from Movia_H, ...
            return share[0], share[1].lower().split('_')[0]
        if isinstance(share[0], tuple):
            return tuple((x[0], x[1].lower().split('_')[0]) for x in share)

        return share

    def share(self):
        """
        Share the zone work between the operators on the trip
        """
        val = tuple(
            (self.stop_sequence,
            self.operator_sequence,
            self.usage_sequence)
            )
        try:
            return self._standardise(self.SHARE_CACHE[val])
        except KeyError:
            pass

        if self.single:
            return self._standardise(self._share_single(val))

        self._remove_cotr()

        if not all(len(set(x)) == 1 for x in self.operator_legs):
            try:
                new_op_legs = self._station_operators()
            except (TypeError, KeyError):
                self.SHARE_CACHE[val] = 'station_map_error'
                return 'station_map_error'
            try:
                self.operator_legs = legops(new_op_legs)
            except IndexError:
                self.SHARE_CACHE[val] = 'operator_error'
                return 'operator_error'

        if not all(x for x in self.operator_legs):
            self.SHARE_CACHE[val] = 'operator_error'
            return 'operator_error'

        self.operator_sequence = tuple(
            [x[0] for x in self.operator_legs] + [self.operator_legs[-1][1]]
            )
        self.single = self._is_single()
        if self.single:
            return self._standardise(self._share_single(val))

        property_dict = self.property_dict()
        property_dict['ops_in_touched'] = operators_in_touched_(
            property_dict['touched_zones'],
            property_dict['zone_legs'],
            self.operator_legs
            )

        shares = self.share_calculation(property_dict)
        self.SHARE_CACHE[val] = shares
        return self._standardise(shares)


    @staticmethod
    def _weight_solo(share_tuple, solo_price_map):
        """weight the shares by the solo zoner pris"""

        if not isinstance(share_tuple[0], tuple):
            share_tuple = (share_tuple, )

        total_zones = round(sum(x[0] for x in share_tuple))
        original_share = tuple(
            (x[0] / total_zones, x[1])
            for x in share_tuple
            )

        operator_prices = tuple(
            (x[0] * solo_price_map[x[1]], x[1])
            for x in share_tuple
            )
        total_price = sum(x[0] for x in operator_prices)

        solo_share = tuple(
            (x[0] / total_price, x[1])
            for x in operator_prices
            )

        weighted_solo = tuple(
            (j[0] * (solo_share[i][0] / original_share[i][0]), j[1])
            for i, j in enumerate(share_tuple)
            )

        return weighted_solo

    def solo_zone_price(self):
        """return the shares weighted by solo zone price"""
        shares = self.share()

        solo_prices = SOLO_ZONE_PRIS[self.region]

        try:
            return self._standardise(self._weight_solo(shares, solo_prices))
        except TypeError:
            return shares
