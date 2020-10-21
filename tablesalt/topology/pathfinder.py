# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:20:27 2019

@author: alkj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:31:40 2019

@author: alkj


functions and Classes used to find
all the zones travelled through on
a trip. Calculated from the given zones
in which a user taps in or out

"""
import pkg_resources
from itertools import chain, groupby
from collections import Counter
from functools import lru_cache
from typing import Tuple, Union, Optional


import pandas as pd
import numpy as np

from tablesalt.common import triptools
from tablesalt.common.io import mappers
from tablesalt.topology import stationoperators


OPGETTER = stationoperators.StationOperators(
    'kystbanen', 'local', 'metro', 'suburban', 'fjernregional'
    )

OP_MAP = {v: k for k, v in mappers['operator_id'].items()}
rev_model_dict = {v:k for k, v in mappers['model_dict'].items()}
CO_TR = (rev_model_dict['Co'], rev_model_dict['Tr'])



# TODO load from config
SOLO_ZONE_PRIS = {
    'th': {
        'dsb': 6.38, 
        'movia': 9.18, 
        'first': 6.42, 
        'stog': 7.12, 
        'metro': 9.44
        }, 
    'ts': {
        'dsb': 6.55, 
        'movia': 7.94, 
        'first': 6.42, 
        'stog': 7.12, 
        'metro': 9.44
        }, 
    'tv': {
        'dsb': 6.38, 
        'movia': 8.43, 
        'first': 6.42, 
        'stog': 7.12, 
        'metro': 9.44
        },
    'dsb': {
        'dsb': 6.57, 
        'movia': 6.36, 
        'first': 6.42, 
        'stog': 7.12, 
        'metro': 9.44
        }
    }

def _determine_region(zone_sequence: Tuple[int, ...]) -> str:
    
    if all(x < 1100 for x in zone_sequence):
        return "th"       
    if all(1100 < x <= 1200 for x in zone_sequence):
        return "tv"
    if all(1200 < x < 1300 for x in zone_sequence):
        return "ts"
    return "dsb"


def load_border_stations(): # put this in TBSTtopology


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
    inborder = {k + 90000: border_dict[k] for k in inborder}
    
    return {**border_dict, **inborder}

BORDER_STATIONS = load_border_stations()

@lru_cache(2**16)
def _is_bus(stopid):
    
    
    return (stopid > stationoperators.MAX_RAIL_UIC or 
            stopid < stationoperators.MIN_RAIL_UIC)

@lru_cache(2**16)
def impute_leg(g, zone_leg):
    """
    for the two touched zones on a zone leg,
    fill in the zones that the leg travels through

    parameters
    ----------
    zone_leg:
        the tuple of zones to impute
    vis_zones:
        the tuple of visited_zones

    """

        
    return g.shortest_path(*zone_leg)[0]

@lru_cache(2**16)
def impute_zone_legs(g, trip_zone_legs):

    return tuple(impute_leg(g, leg)
                 for leg in trip_zone_legs)

@lru_cache(2**16)
def get_touched_zones(zone_sequence) -> Tuple[int, ...]:
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
def to_legs(sequence):

    return triptools.sep_legs(sequence)

@lru_cache(2**16)
def _to_legs_stops(stop_legs):
    return tuple(
            (leg[0], OPGETTER.BUS_ID_MAP.get(leg[1])) if 
            (not _is_bus(leg[0]) and _is_bus(leg[1])) else leg 
            for leg in stop_legs
            )     

@lru_cache(2**16)
def _chain_vals(vals):
    
    lvals = len(vals)
    visited_zones = tuple(
        chain(*[j[:-1] if i != lvals - 1 else j
              for i, j in enumerate(vals)])
        )    
    return visited_zones

def _border_legs():
    
    return 

class ZoneProperties():

    
    VISITED_CACHE = {}

    def __init__(self, 
                 graph, 
                 zone_sequence: Tuple[int, ...], 
                 stop_sequence: Tuple[int, ...],
                 region: Optional[str] = 'sjælland') -> None:
        
        """


        Parameters
        ----------
        graph : TYPE
            DESCRIPTION.
        zone_sequence : TYPE
            DESCRIPTION.
        stop_sequence : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.graph = graph
        # self.ring_dict = self.graph.ring_dict(region)
        self.zone_sequence: Tuple[int, ...] = zone_sequence
        self.stop_sequence: Tuple[int, ...]  = stop_sequence
               
        self.stop_legs: Tuple[Tuple[int, ...], ...] = to_legs(stop_sequence)
        self.stop_legs = _to_legs_stops(self.stop_legs)
        
        self.zone_legs: Tuple[Tuple[int, ...], ...] = to_legs(zone_sequence)
        
        self.border_trip: bool = False
        self.border_legs: Union[Tuple[()], Tuple[int, ...]]  = ()
        
        if any(x in BORDER_STATIONS for x in chain(*self.stop_legs)):
            self.border_trip = True

        if self.border_trip:
            self.border_legs = self._border_touch_legs()

        self.touched_zones = get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = to_legs(self.touched_zones)
    
    def _border_touch_legs(self) -> Tuple[int, ...]:
        """
        return the card touch positions on the trip
        as a tuple
        """

        return tuple(i for i, j in enumerate(self.stop_legs)
                     if any(x in BORDER_STATIONS for x in j))


    def _visited_zones_on_leg(self, zone_leg: Tuple[int, int]) -> None:
        """
        for a tuple of start and end zone
        find the zones travelled through
        and update the cache
        """

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

    def get_visited_zones(self, zone_legs) -> Tuple[Union[int, Tuple[int, ...]], ...]:
        """
        visited zones is a list of the zones visited on a trip,
        in order, but removing adjacent duplicate zones

        """

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

     
    def _borderless_properties(self): 
        
        visited_zones = self.get_visited_zones(self.touched_zone_legs)
        
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

        double_back = bool(max(Counter(chained_zones).values()) > 1)

        prop_dict = {
            'visited_zones': visited_zones,
            'total_travelled_zones': len(visited_zones),
            'double_back': double_back,
            'touched_zones': self.touched_zones,
            'stop_sequence': self.stop_sequence,
            'stop_legs': self.stop_legs,
            'zone_sequence': self.zone_sequence,
            'zone_legs': self.zone_legs,
            'border_legs': self.border_legs
            }        
        
        return prop_dict
    
    
    def _bordered_properties(self):
                       
        zone_legs = []
        for legnr, zone_leg in enumerate(self.zone_legs):                   
            start_zone = zone_leg[0]
            end_zone = zone_leg[1]
            if legnr in self.border_legs: 
                stop_leg = self.stop_legs[legnr]                
                if stop_leg[0] in BORDER_STATIONS:
                    border_zones = BORDER_STATIONS[stop_leg[0]] 
                    border_distances = [
                        len(self.graph.shortest_path(x, zone_leg[1])[0])
                        for x in border_zones
                        ]
                    min_pos = np.argmin(border_distances)
                    start_zone = border_zones[min_pos]
                if stop_leg[1] in BORDER_STATIONS:
                    border_zones = BORDER_STATIONS[stop_leg[1]] 
                    border_distances = [
                            len(self.graph.shortest_path(zone_leg[0], x)[0])
                            for x in border_zones
                            ]
                    min_pos = np.argmin(border_distances)
                    end_zone = border_zones[min_pos]
            
            leg = (start_zone, end_zone)
            zone_legs.append(leg)
                
        self.zone_sequence = tuple(
            [x[0] for x in zone_legs] + [zone_legs[-1][1]]
            )
        self.zone_legs = to_legs(self.zone_sequence)   
        self.touched_zones = get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = to_legs(self.touched_zones)
                
        return self._borderless_properties()

    def property_dict(self):

        if not self.border_trip:
            return self._borderless_properties()
        
        return self._bordered_properties()



def _remove_idxs(idxs, legs):
    
    
    return tuple(j for i, j in enumerate(legs) if i not in idxs) 

def legops(new_legs):
    """
    just legify the output from determin_operator
    """

    out = []
    for x in new_legs:
        if len(x) == 1:
            out.append((x[0], x[0]))
        else:
            out.append((x[0], x[1]))
    return tuple(out)

def operators_in_touched_(tzones, zonelegs, oplegs):
    """
    determine the operators in the zones that are touched
    returns a dictionary:

    parameters
    -----------
    tzones:
        a tuple of the zones touched by
        a rejsekort tap on a trip
    zonelegs:
        a legified tuple of zones
    oplegs:
        a legified tuple of operators

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
            graph, 
            zone_sequence: Tuple[int, ...],  
            stop_sequence: Tuple[int, ...],              
            operator_sequence: Tuple[int, ...], 
            usage_sequence: Tuple[int, ...]
            ) -> None:
        super().__init__(graph, zone_sequence, stop_sequence)
        
        self.zone_sequence = zone_sequence
        self.stop_sequence = stop_sequence
        self.operator_sequence = operator_sequence
        self.usage_sequence = usage_sequence        
        self.operator_legs = to_legs(self.operator_sequence)
        self.usage_legs = to_legs(self.usage_sequence)
        
        self.single = self._is_single()
        
        self.region: str = _determine_region(self.zone_sequence)

    
    def _is_single(self) -> bool:
        
        return len(set(chain(*self.operator_legs))) == 1
    
    def _remove_cotr(self) -> None:
        
        if CO_TR not in self.usage_legs:
            return 
        CoTr_idxs = ()
        for i, j in enumerate(self.usage_legs):
            if j == CO_TR:
                CoTr_idxs += (i,)
        
        self.stop_legs = _remove_idxs(CoTr_idxs, self.stop_legs)
        self.operator_legs = _remove_idxs(CoTr_idxs, self.operator_legs)
        self.zone_legs = _remove_idxs(CoTr_idxs, self.zone_legs)
        self.usage_legs = _remove_idxs(CoTr_idxs, self.usage_legs)

    def _station_operators(self):
        
        oplegs = tuple(
            OPGETTER.station_pair(*x, format='operator_id') 
            for x in self.stop_legs
                )
            
        return oplegs
    
    def _share_single(self, val):
        shares = (
            self.property_dict()['total_travelled_zones'], 
            OP_MAP[self.operator_sequence[0]]
            )
        self.SHARE_CACHE[val] = shares
        return shares
    
    def share_calculation(self, val):
        """
        calculate the shares for
        """
      
        out = {}
        # zone_counts = Counter(val['visited_zones'])
    
        for i, imputed_leg in enumerate(val['imputed_zone_legs']):
            for zone in imputed_leg:
                if zone in val['nlegs_in_touched']:
                    if val['nlegs_in_touched'][zone] == 1:
                        out[zone] = 1, self.operator_legs[i][0]
                    else:
                        counts = Counter(val['ops_in_touched'][zone])
                        
                        try:
                            out[zone] = tuple((v/val['nlegs_in_touched'][zone], k) for
                                              k, v in counts.items())
                        except ZeroDivisionError:
                            out[zone] = 1, self.operator_legs[i][0]
                else:
                    out[zone] = 1, self.operator_legs[i][0]
        
        return aggregated_zone_operators(tuple(out.values()))
    
    @staticmethod
    def _standardise(share):
        
        if isinstance(share[0], int):
            return share[0], share[1].lower().split('_')[0]
        if isinstance(share[0], tuple):
            return tuple((x[0], x[1].lower().split('_')[0]) for x in share)
     
        return share
    
    def share(self):
        
        val = tuple(
            (self.stop_sequence, self.operator_sequence, self.usage_sequence)
            )
        try:
            return self._standardise(self.SHARE_CACHE[val]) 
        except KeyError:
            pass
        
        if self.single:
            return self._standardise(self._share_single(val))

        
        self._remove_cotr()
        
        if not all(len(set(x))== 1 for x in self.operator_legs):
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
        property_dict['nlegs_in_touched'] = {
            tzone: len([x for x in property_dict['zone_legs'] if tzone in x])
            for tzone in property_dict['touched_zones']
            }
        property_dict['ops_in_touched'] = operators_in_touched_(
            property_dict['touched_zones'],
            property_dict['zone_legs'], 
            self.operator_legs
            )
        property_dict['imputed_zone_legs'] = \
        impute_zone_legs(self.graph, property_dict['zone_legs'])
        
        shares = self.share_calculation(property_dict)
        self.SHARE_CACHE[val] = shares
        return self._standardise(shares)
    
    
    @staticmethod
    def _weight_solo(share_tuple, solo_price_map):
        
        if not isinstance(share_tuple[0], tuple):
            share_tuple = (share_tuple, )            
        
        total_zones = round(sum(x[0] for x in share_tuple))
        original_share = tuple(
            (x[0] / total_zones, x[1]) 
            for x in share_tuple)
        
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
        
        shares = self.share()
        
        solo_prices = SOLO_ZONE_PRIS[self.region]
        
        try:           
            return self._weight_solo(shares, solo_prices)
        except TypeError:
            return shares

def shrink_search(graph, start, goal, ringzones, distance_buffer=2):
    """
    subset the graph based on a given distance_buffer of zones
    Not used yet

    parameters
    -----------
    graph:
        a networkx graph of zones
    start:
        the start zone in national format
    goal:
        the desired end zone in national format
    ringzones:
        the ringzone dictionary
    distance_buffer:
        a number of zones to buffer around the search path
        int
        default=2


    """

    distance = ringzones[(start, goal)]
    distance += distance_buffer
    sub_rings = {k:v for k, v in ringzones.items()
                 if k[0] == start and v <= distance}
    sub_zones = set(chain(*sub_rings.keys()))
    new_graph = {k:v for k, v in graph.items() if
                 k in sub_zones}
    new_graph = {k:set(x for x in v if x in sub_zones) for
                 k, v in new_graph.items()}

    return new_graph

