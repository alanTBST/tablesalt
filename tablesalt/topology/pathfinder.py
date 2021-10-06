# -*- coding: utf-8 -*-
"""
Functions and Classes used to find
all the zones travelled through on
a trip. Calculated from the given zones
in which a user taps in or out

"""

from itertools import chain, groupby
from collections import Counter, defaultdict
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


OPGETTER = stationoperators.StationOperators(
    'kystkastrup', 'sjællandlocal', 'metro',
    'suburban', 'sjællandfjernregional',
    )

OP_MAP = {v: k.lower() for k, v in mappers['operator_id'].items()}
REV_OP_MAP = {v: k for k, v in OP_MAP.items()}

rev_model_dict = {v:k for k, v in mappers['model_dict'].items()}
CO_TR = (rev_model_dict['Co'], rev_model_dict['Tr'])
SU_SU = (rev_model_dict['Su'], rev_model_dict['Su'])


METRO_MAP = mappers['metro_map']
REV_METRO_MAP = mappers['metro_map_rev']

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

@lru_cache(2**8)
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

@lru_cache(2**8)
def _to_legs_stops(stop_legs):
    """convert to legs and map the bus id to station if possible"""
    return tuple(
            (leg[0], OPGETTER.BUS_ID_MAP.get(leg[1])) if
            (not _is_bus(leg[0]) and _is_bus(leg[1])) else leg
            for leg in stop_legs
            )

@lru_cache(2**8)
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

@lru_cache(2**8)
def aggregated_zone_operators(v):
    "aggregate all the operators for the zone shares"

    vals = list(v)

    out_list = []
    for x in vals:
        if isinstance(x[0], (int, float)):
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



STOPS = StopsList.default_denmark().stops_dict

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
        self.stops_dict = STOPS

        self.region: str = determine_takst_region(self.zone_sequence)


    def _is_single(self) -> bool:
        """has only one operator"""
        return len(set(chain(*self.operator_legs))) == 1

    def _remove_cotr(self) -> None:
        """Remove all of the legs that are cotr touches
        for each relevant attribute
        """
        # NOTE...THIS can be put in TRIPRECORD

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

    def _remove_susu(self) -> None:
        """Remove SuSu legs if they occur at the same station
        """
        # NOTE...THIS can be put in TRIPRECORD


        susu_idxs = ()
        for i, j in enumerate(self.stop_legs):
            if j[0] == j[1]:
                susu_idxs += (i,)
                continue
            if j[0] == METRO_MAP.get(j[1]) or j[1] == METRO_MAP.get(j[0]):
                susu_idxs += (i,)
                continue
            start = self.stops_dict[j[0]]
            end =  self.stops_dict[j[1]]
            if j[0] in end.alternate_stop_ids or j[1] in start.alternate_stop_ids:
                susu_idxs += (i,)
        if susu_idxs:
            self.stop_legs = _remove_idxs(susu_idxs, self.stop_legs)
            self.operator_legs = _remove_idxs(susu_idxs, self.operator_legs)
            self.zone_legs = _remove_idxs(susu_idxs, self.zone_legs)
            self.usage_legs = _remove_idxs(susu_idxs, self.usage_legs)

    def _station_operators(self) -> Tuple[int, ...]:
        """Get the operators at the visited stations"""

        opsequence = ()
        for x in self.stop_legs:
            ops = OPGETTER.station_pair(*x, format='operator')
            op = stationoperators.REV_OP_MAP[(list(ops)[0])]
            opsequence += (op, )

        return opsequence

    @staticmethod
    def leg_solo_price(region: str, op_id: int) -> float:
        """Find the solo price of the leg for the specified
        operator id

        :param region: the region the leg is in
        :type region: str
        :param op_id: the operator id of the leg
        :type op_id: int
        :return: a solo zone price
        :rtype: float
        """
        return SOLO_ZONE_PRIS[region][OP_MAP[op_id]]

    def _single_zone_share(
        self,
        zone: int,
        op_id: int,
        touched_zones,
        touched_zone_leg_count,
        ops_in_touched_zones
        ):


        zone_share = 1
        if zone in touched_zones:
            if touched_zone_leg_count[zone] == 1:
                res = (zone_share, op_id)
            else:
                # need to assign the fraction of the zone for the leg based on it's region for solo
                n_legs_in_zone = touched_zone_leg_count[zone]
                n_op_id_legs_in_zone = Counter(ops_in_touched_zones[zone])[op_id]
                try:
                    zone_share = n_op_id_legs_in_zone / n_legs_in_zone
                    res = (zone_share, op_id)
                except ZeroDivisionError:
                    res = (zone_share, op_id)
        else:
            res = (zone_share, op_id)

        return res

    @staticmethod
    def _weight_solo(disaggregate_zones, solo_price_legs):
        """weight the shares by the solo zoner pris"""
        combined = list(zip(chain(*disaggregate_zones),
                            chain(*solo_price_legs)))


        total_zones = round(sum(x[0][0] for x in combined))
        total_price = sum(x[0][0]*x[1] for x in combined)

        out = []
        for zone_operator, soloprice in combined:
            zone, operator = zone_operator
            val = ((zone * soloprice) / total_price) * total_zones
            out.append((val, operator))
        return aggregated_zone_operators(tuple(out))

    @staticmethod
    def _weight_bumped_zones(out_standard, out_bumped, min_zones):

        outstandard = list(chain(*out_standard.values()))
        sum_standard = sum(x[0] for x in outstandard)
        # bumping must be minimum two zones on total (this a)
        sum_standard = sum_standard if sum_standard >= min_zones else min_zones
        sum_bumped = sum(x[0] for x in out_bumped)

        final_bump = [(x[0] / sum_bumped * sum_standard, x[1])
                      for x in out_bumped]

        return final_bump

    @staticmethod
    def _weight_solo_bumped(out_bumped, out_bumped_solo):
        """weight the shares by the solo zoner pris"""

        out_bumped_solo = [x[0] for x in out_bumped_solo]
        combined = list(zip(out_bumped, out_bumped_solo))

        total_zones = round(sum(x[0][0] for x in combined))
        total_price = sum(x[0][0]*x[1] for x in combined)

        out = []
        for zone_operator, soloprice in combined:
            zone, operator = zone_operator
            val = ((zone * soloprice) / total_price) * total_zones
            out.append((val, operator))
        return aggregated_zone_operators(tuple(out))

    def share_calculation(
        self,
        properties: Dict[str, Any],
        min_zones: Optional[int] = 2
        ) -> Union[Tuple[int, str], Tuple[Tuple[int, str], ...]]:
        """
        Calculate the zone shares for the operators on the trip

        :param val: property_dict from ZoneProperties
        :type val: Dict[str, Any]
        :return: the zone shares for the trip
        :rtype: Union[Tuple[int, str], Tuple[Tuple[int, str], ...]]

        """
        imputed_zone_legs = properties['imputed_zone_legs']
        zone_leg_regions = properties['zone_legs_regions']

        touched_zones = properties['touched_zones']
        touched_zone_leg_count = properties['nlegs_in_touched']
        ops_in_touched_zones = properties['ops_in_touched']

        out_standard = defaultdict(list)
        out_solo = defaultdict(list)

        out_bumped = []
        out_bumped_solo = []

        current_op_sum = 0
        previous_op_id = None

        operatorshift_flag = False

        seen_zone_opid = set()
        for legnum, imputed_leg in enumerate(imputed_zone_legs):

            leg_region = zone_leg_regions[legnum]
            op_id = self.operator_legs[legnum][0]
            leg_solo_price = self.leg_solo_price(leg_region, op_id)

            if not op_id == previous_op_id and previous_op_id is not None:
                current_op_sum = 0

            for zone in imputed_leg:
                if (zone, op_id) not in seen_zone_opid:
                    res = self._single_zone_share(
                        zone, op_id, touched_zones,
                        touched_zone_leg_count,
                        ops_in_touched_zones
                        )
                    out_standard[zone].append(res)
                    out_solo[zone].append(leg_solo_price)
                    current_op_sum += res[0]
                    seen_zone_opid.add((zone, op_id))

            try:
                next_op_id = self.operator_legs[legnum+1][0]
                if op_id != next_op_id:
                    if current_op_sum < min_zones:
                        current_op_sum = min_zones
                    operatorshift_flag = True
            except IndexError:
                if current_op_sum < min_zones:
                    current_op_sum = min_zones
                operatorshift_flag = True

            if operatorshift_flag:
                out_bumped.append((current_op_sum, op_id))
                out_bumped_solo.append((leg_solo_price, op_id))

            previous_op_id = op_id

        out_standard = {k: tuple(v) for k, v in out_standard.items()}

        standard = aggregated_zone_operators(tuple(out_standard.values()))

        if self.single:
            solo_price = standard
        else:
            solo_price = self._weight_solo(out_standard.values(), out_solo.values())

        if self.single:
            bumped = out_bumped
            bumped_solo = out_bumped
        else:
            bumped = self._weight_bumped_zones(out_standard, out_bumped, min_zones)
            bumped_solo = self._weight_solo_bumped(bumped, out_bumped_solo)

        bumped_out = aggregated_zone_operators(tuple(bumped))
        try:
            bumped_solo = aggregated_zone_operators(tuple(bumped_solo))
        except KeyError:
            pass

        return {'standard': standard,
                'solo_price': solo_price,
                'bumped': bumped_out,
                'bumped_solo': bumped_solo}

    @staticmethod
    def _error_shares(error_name: str) -> Dict[str, str]:
        """if there is an error, create an output dictionary
        for it

        :param error_name: the name of the error
        :type error_name: str
        :return: a dicionary with an error for each model
        :rtype: Dict[str, str]
        """

        return {'standard': error_name,
                'solo_price': error_name,
                'bumped': error_name,
                'bumped_solo': error_name}

    def share(
        self,
        minimum_operator_zones: Optional[int] = 2
        ) -> Dict[str, Tuple[Tuple[Union[int, float], str], ...]]:
        """[summary]

        :param minimum_operator_zones: The number of zones in the
            minimum principle model, defaults to 2
        :type minimum_operator_zones: Optional[int], optional
        :return: [description]
        :rtype: Dict[str, Tuple[Tuple[Union[int, float], str], ...]]
        """


        val = tuple(
            (self.stop_sequence,
            self.operator_sequence,
            self.usage_sequence)
            )
        if None in chain(*self.stop_legs):
            shares = self._error_shares('station_map_error')
            self.SHARE_CACHE[val] = shares
            return shares

        try:
            return self.SHARE_CACHE[val]
        except KeyError:
            pass

        self._remove_cotr()
        try:
            self._remove_susu()
        except KeyError:
            shares = self._error_shares('station_map_error')
            self.SHARE_CACHE[val] = shares
            return shares

        if not all(len(set(x)) == 1 for x in self.operator_legs):
            try:
                new_op_legs = self._station_operators()
            except (ValueError, TypeError):
                shares = self._error_shares('station_map_error')
                self.SHARE_CACHE[val] = shares
                return shares
            except IndexError:
                shares = self._error_shares('operator_error')
                self.SHARE_CACHE[val] = shares
                return shares

            self.operator_legs = tuple((x, x) for x in new_op_legs)

        if not all(len(set(x)) == 1 for x in self.operator_legs):
            shares = self._error_shares('operator_error')
            self.SHARE_CACHE[val] = shares
            return shares

        try:
            properties = self.property_dict()
        except IndexError:
             # all legs removed
            shares = self._error_shares('no_available_trip')
            self.SHARE_CACHE[val] = shares
            return shares

        properties['ops_in_touched'] = operators_in_touched_(
            properties['touched_zones'],
            properties['zone_legs'],
            self.operator_legs
            )

        try:
            shares = self.share_calculation(properties, minimum_operator_zones)
        except KeyError:
            shares = self._error_shares('rk_operator_error')
            self.SHARE_CACHE[val] = shares
            return shares

        self.SHARE_CACHE[val] = shares
        return shares
