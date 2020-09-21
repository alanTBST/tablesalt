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
from itertools import chain
from collections import Counter
from typing import Tuple, Union, Optional


import pandas as pd
import numpy as np

from tablesalt.common import triptools
from tablesalt.common.io import mappers
from tablesalt.topology import stationoperators


OPGETTER = stationoperators.StationOperators(
    'kystbanen', 'local', 'metro', 'suburban', 'fjernregional'
    )


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


def _is_bus(stopid):
    
    
    return (stopid > stationoperators.MAX_RAIL_UIC or 
            stopid < stationoperators.MIN_RAIL_UIC)


def impute_leg(zone_leg, vis_zones):
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

    vis_zones_ = list(vis_zones)

    visited_idxs = [vis_zones_.index(int(zone)) for zone in zone_leg]

    return vis_zones[visited_idxs[0]: visited_idxs[1] + 1]


def impute_zone_legs(trip_zone_legs, visited_zones):

    return tuple(impute_leg(leg, visited_zones)
                 for leg in trip_zone_legs)



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
        self.ring_dict = self.graph.ring_dict(region)
        self.zone_sequence: Tuple[int, ...] = zone_sequence
        self.stop_sequence: Tuple[int, ...]  = stop_sequence
               
        self.stop_legs: Tuple[Tuple[int, ...], ...] = self._to_legs(stop_sequence)
        self.stop_legs = tuple(
            (leg[0], OPGETTER.BUS_ID_MAP.get(leg[1])) if 
            (not _is_bus(leg[0]) and _is_bus(leg[1])) else leg 
            for leg in self.stop_legs
            )
        
        self.zone_legs: Tuple[Tuple[int, ...], ...] = self._to_legs(zone_sequence)
        
        self.border_trip: bool = False
        self.border_legs: Union[Tuple[()], Tuple[int, ...]]  = ()
        
        if any(x in BORDER_STATIONS for x in chain(*self.stop_legs)):
            self.border_trip = True

        if self.border_trip:
            self.border_legs = self._border_touch_legs()

        self.touched_zones = self.get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = self._to_legs(self.touched_zones)
    
    def _border_touch_legs(self) -> Tuple[int, ...]:
        """
        return the card touch positions on the trip
        as a tuple
        """

        return tuple(i for i, j in enumerate(self.stop_legs)
                     if any(x in BORDER_STATIONS for x in j))

    def get_touched_zones(self, zone_sequence) -> Tuple[int, ...]:
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
    
    @staticmethod
    def _to_legs(sequence):

        return triptools.sep_legs(sequence)

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

        lvals = len(vals)
        visited_zones = tuple(
            chain(*[j[:-1] if i != lvals - 1 else j
                  for i, j in enumerate(vals)])
            )

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

        counts = Counter(chained_zones)
        double_back = bool(max(counts.values()) > 1)

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
        
        # border_count = len({x for x in self.stop_sequence if x in BORDER_STATIONS})
        
        
        zone_seq = []
        seen_zones = set()
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
                if start_zone not in seen_zones:
                    zone_seq.append(start_zone)   
                if end_zone not in seen_zones:         
                    zone_seq.append(end_zone)
                
            seen_zones.add(start_zone) 
            seen_zones.add(end_zone)
        self.zone_sequence = tuple(zone_seq)
        self.zone_legs = self._to_legs(self.zone_sequence)   
        self.touched_zones = self.get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = self._to_legs(self.touched_zones)
                
        return self._borderless_properties()
    @property
    def property_dict(self):
        """
        return the three properties of the zone sequnce
        in a dictionary.
        The dictionary contains:
            'visited_zones' -
            'total_travelled_zones' -
            'double_back' -
        """
        if not self.border_trip:
            return self._borderless_properties()
        
        return self._bordered_properties()


class ZoneSharer:
    
    def __init__(
            self, 
            zone_properties,
            operator_sequence: Tuple[int, ...], 
            usage_sequence: Tuple[int, ...]
            ) -> None:
        
        self.zone_properties = zone_properties
        self.operator_sequence = operator_sequence
        self.usage_sequence = usage_sequence
        
        self.single = self._is_single()    
    
    def _is_single(self):
        
        return len(set(self.operator_sequence)) == 1
    
    def share(self):
        shares = ()
        return shares
    
    
def operators_in_touched_(tzones, zonelegs, oplegs, border_zones=None):
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
    if not border_zones:
        return ops_in_touched

    return {k:v if k not in border_zones else (v[0],)
            for k, v in ops_in_touched.items()}


def aggregated_zone_operators(vals):
    """

    perform the aggregation

    parameter
    ---------
    vals:
        a list or tuple of tuples of the form:
            ((n_zones[0], op_id[0]), (n_zones[1], op_id[1]),....(n_zones[n], op_id[n]))
    """

    vals = list(vals.values())

    out_list = []
    for x in vals:
        if isinstance(x[0], int):
            out_list.append(x)
            continue
        for y in x:
            out_list.append(y)

    out_list = sorted(out_list, key=lambda x: x[1])

    return tuple(((sum(x[0] for x in grp), key)) for
                 key, grp in groupby(out_list, key=lambda x: x[1]))

def needs_assignment_check(oplegs):
    """
    return boolean
    True if there needs to be an operator check
    False otherwise
    """
    return any(len(set(leg)) > 1 for leg in oplegs)

def removeCoTr(val, CoTr):
    """
    remove the checkout - checkin again (CoTr) legs
    """
    # (2, 4) corresponds to (Co, Tr) currently
    if CoTr not in val['usage_legs']:
        return val
    CoTr_idxs = []
    for i, j in enumerate(val['usage_legs']):
        if j == CoTr:
            CoTr_idxs.append(i)
    
    new_val = val.copy()
    for x in ('stop_legs', 'op_legs', 'zone_legs', 'border_legs'):
        try:
            new_val[x] = tuple(
                j for i, j in enumerate(new_val[x]) if i not in CoTr_idxs
                )
        except (KeyError, ValueError, TypeError):
            pass
    return new_val


def contains_border(stoplegs, border_stations):
    """
    parameters
    ----------
    stoplegs:
        tuple of legified stop id legs
    border_stations:
        dict of border stations
    returns boolean
    True if
    """

    return any(x in border_stations for x in chain(*stoplegs))

def _no_borders(val):

    val['nlegs_in_touched'] = {
        tzone: len([x for x in val['zone_legs'] if tzone in x])
        for tzone in val['touched_zones']
        }
    val['ops_in_touched'] = operators_in_touched_(
        val['touched_zones'], val['zone_legs'], val['new_op_legs']
        )
    val['imputed_zone_legs'] = \
    impute_zone_legs(val['zone_legs'], val['visited_zones'])

    return val


def _with_borders(val, border_stations):

    val['imputed_zone_legs'] = \
    impute_zone_legs(val['zone_legs'], val['visited_zones'])

    nlegs_in_touched = {}

    bstations = set(border_stations)
    bleg_stops = set(chain(*[j for i, j in enumerate(val['stop_legs'])
                     if i in val['border_legs']]))
    try:
        border_station_zones = \
        border_stations[list(bstations.intersection(bleg_stops))[0]]
    except IndexError:
        border_station_zones = []

    for tzone in val['touched_zones']:
        tzone_count = 0
        for i, j in enumerate(val['zone_legs']):
            if tzone in j:
                if i not in val['border_legs'] or \
                tzone not in border_station_zones:
                    tzone_count += 1
                elif i in val['border_legs']:
                    imputed = val['imputed_zone_legs'][i]
                    if all(x in imputed for x in border_station_zones):
                        tzone_count += 1

        nlegs_in_touched[tzone] = tzone_count
    val['nlegs_in_touched'] = nlegs_in_touched

    val['ops_in_touched'] = operators_in_touched_(
        val['touched_zones'], val['zone_legs'], val['new_op_legs'],
        border_zones=border_station_zones)

    return val

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



def procprop(properties, border_stations):
    """
    process_properties frim multisharing
    """
    rev_model_dict = {v:k for k, v in mappers['model_dict'].items()}
    co_tr_tuple = (rev_model_dict['Co'], rev_model_dict['Tr'])
    # missed_check_point = []
    # output = []
    # op_erros = []
    # other_errors = []
    for val in properties:
        try:
            val = removeCoTr(val, co_tr_tuple)
            bordercheck = contains_border(val['stop_legs'], border_stations)
            try:
                new_legs = tuple(
                    OPGETTER.station_pair(*x, format='operator') for x in val['stop_legs']
                    )
                if not all(x for x in new_legs):
                    # missed_check_point.append(val)
                    continue
                val['new_op_legs'] = legops(new_legs)
            except (KeyError, ValueError):
                # op_erros.add(val['tripkey'])
                continue
 
            if bordercheck:
                val = _with_borders(val, border_stations)
            else:
                val = _no_borders(val)
            yield val

        except Exception:
            # other_errors.append(val)
            continue


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

