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


def load_border_stations(): # put this in TBSTtopology


    fp = pkg_resources.resource_filename(
        'tablesalt',
        'resources/revenue/borderstations.xlsx'
        )
    border_frame = pd.read_excel(fp)
    border_frame['Zones'] = list(zip(
        border_frame['Zone1'], border_frame['Zone2'], border_frame['Zone3']))
    return {k:tuple(int(x) for x in v if x > 0)
        for k, v in border_frame.set_index('UIC')['Zones'].to_dict().items()}

BORDER_STATIONS = load_border_stations()

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


def _leg_borders(stop_sequence, border_positions):
    
    leg_border_dict = {}
    
    for i, j in enumerate(stop_sequence):
        if i == 0:
            continue    
        if i in border_positions:
            leg_border_dict[i-1] = 1
            leg_border_dict[i] = 0
               
    return leg_border_dict
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
        self.zone_sequence = zone_sequence
        self.stop_sequence = stop_sequence
        self.stop_legs = self._to_legs(stop_sequence)
        self.border_trip = False
        self.border_positions = None
        self.border_legs = None
        
        if any(x in BORDER_STATIONS for x in stop_sequence):
            self.border_trip = True

        if self.border_trip:
            self.border_positions = self._border_touch_postions()
            self.border_legs = self._border_touch_legs()

        self.touched_zones = self.get_touched_zones(self.zone_sequence)
        self.touched_zone_legs = self._to_legs(self.touched_zones)
    
    def _border_touch_postions(self) -> Tuple[int, ...]:
        """
        return the card touch positions on the trip
        as a tuple
        """

        return tuple(i for i, j in enumerate(self.stop_sequence)
                     if j in BORDER_STATIONS)
    
    def _border_touch_legs(self) -> Tuple[int, ...]:
        """
        return the card touch positions on the trip
        as a tuple
        """

        return tuple(i for i, j in enumerate(self.stop_legs)
                     if any(x in BORDER_STATIONS for x in j))
    
    def _border_is_dest(self) -> bool:
        """
        test whether a border station check is the
        ultimate check
        returns boolean
        """
        assert self.border_trip
        return self.border_positions[-1] == len(self.stop_sequence) - 1

    def _border_is_orig(self) -> bool:
        """
        test whether a border station check is the
        original check
        returns boolean
        """
        assert self.border_trip
        return self.border_positions[0] == 0

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
        for x in zone_legs:
            try:
                vals.append(self.VISITED_CACHE[x])
            except KeyError:
                self._visited_zones_on_leg(x)
                vals.append(self.VISITED_CACHE[x])

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
        # must put in alternate visited zones for border stations 
        # and choose the minimum
        
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
            'zone_sequence': self.zone_sequence,
            'border_legs': self.border_legs}        
        
        return prop_dict
    
    def _bordered_properties(self):
        prop_dict = self._borderless_properties()
        
        stoplegs = self._to_legs(self.stop_sequence)
        border_legs = [
            i for i, j in enumerate(stoplegs) if
            any(x in BORDER_STATIONS for x in j)
            ]
        
        if self._border_is_dest(): #and not double_back:

            stopnum = self.stop_sequence[-1]
            end_stop_border = BORDER_STATIONS[stopnum]
            if all(x in prop_dict['visited_zones'] for x in end_stop_border):
                vis_list = list(prop_dict['visited_zones'])

        
        return 
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

        both = self._border_is_dest() and self._border_is_orig()
        prop_dict = self._borderless_properties()              
        if self._border_is_dest(): #and not double_back:

            stopnum = self.stop_sequence[-1]
            end_stop_border = BORDER_STATIONS[stopnum]
            if all(x in prop_dict['visited_zones'] for x in end_stop_border):
                vis_list = list(prop_dict['visited_zones'])
                rem = vis_list.pop()
                repl = [x for x in end_stop_border if x != rem][0]
                repld = {rem: repl}
                prop_dict['visited_zones'] = tuple(vis_list)
                prop_dict['total_travelled_zones'] = len(prop_dict['visited_zones'])
                prop_dict['touched_zones'] = tuple(repld.get(x, x) for x in prop_dict['touched_zones'])
                prop_dict['zone_sequence'] = tuple(repld.get(x, x) for x in self.zone_sequence)

        if not self._border_is_dest() and not self._border_is_orig():
            stoplegs = self._to_legs(self.stop_sequence)
            prop_dict['border_legs'] = \
            [i for i, j in enumerate(stoplegs) if
             any(x in BORDER_STATIONS for x in j)]

        touched = set(prop_dict['touched_zones'])
        visited = set(prop_dict['visited_zones'])
        
        if touched != touched.intersection(visited):
            if not touched.difference(visited).intersection(visited):
                prop_dict['touched_zones'] = prop_dict['visited_zones']
            else:
                prop_dict['touched_zones'] = \
                tuple(touched - touched.difference(visited))
       
        if not all(isinstance(x, (np.int64, int)) for x in tuple(prop_dict['touched_zones'])):
            raise TypeError("can't determine touched zones")

        return prop_dict
