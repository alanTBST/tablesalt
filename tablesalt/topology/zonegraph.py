# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:44:06 2020

@author: alkj

Class to create an undirected graph
of the tariffzones in Denmark
"""
#standard imports
import os
import pkg_resources
from itertools import groupby, chain
from collections import Counter
from operator import itemgetter
from typing import Optional


#third party imports
import pandas as pd
import numpy as np
import networkx as nx

# package imports
from tablesalt.topology.tools import TakstZones, EdgeMaker


REGIONS = {
    'hovedstaden',
    'sjælland',
    'vestsjælland',
    'sydsjælland',
    # 'fyn',
    # 'jylland',
    # 'nordjylland',
    # 'midtjylland',
    # 'sydjylland',
    # 'sydjyllandvest',
    # 'sydjyllandøst',
    # 'sydjyllandsyd'
    }

def _determine_region_zones():
    """determine the zones that are in each region from shp"""

    return

REGION_ZONES = {
    'hovedstaden': (1000, 1100), # 200
    'vestsjælland': (1100, 1200), # 230
    'sydsjælland': (1200, 1300), # 235
    'sjælland': (1000, 1300),
    # 'fyn': (2000, 2300), # 242
    # 'fynøst': (2200, 2300), # 242
    # 'fynvest': (2100, 2200), # 242
    # 'fynmidt': (2000, 2100), # 242
    # 'jylland': (3000, 6000),
    # 'nordjylland': (5000, 6000), # 280
    # 'midtjylland': (4000, 5000),
    # 'midtjyllandøst': (4300, 4400), # 270 + 4801-15, 4857-4888 4500-4600
    # 'midtjyllandvest': (4400, 4500), # 265 + 4901, 4902, 4903
    # 'midtjyllandmidt': (4200, 4300), # 276
    # 'sydjylland': (3000, 3800),
    # 'sydjyllandvest': (3600, 3700),  # 255
    # 'sydjyllandøst': (3500, 3600), # 260
    # 'sydjyllandsyd': (3700, 3800),  # 250
    # 'bornholm': (6000, 6100) # 240
    }


def _legify(v):
    # TODO : put legify in TBSTtrips? TBSTtools TBSTutils?
    return tuple((v[i], v[i+1]) for i in range(len(v)-1))


def _neighbour_dict(region):
    """Load and convert neighbours dset to dict (adj list)"""

    fp = pkg_resources.resource_filename(
        'tablesalt', os.path.join('resources', 'networktopodk', 'national_neighbours.csv')
        )
    neighbours = pd.read_csv(
        fp,
        header=None, index_col=0
        )

    zone_min, zone_max = REGION_ZONES[region]

    neighbours = neighbours.fillna(0)
    neighbours = neighbours.astype(int)
    neighbours.loc[:, 'tup'] = neighbours.apply(tuple, axis=1)
    neighbours_dict = neighbours.loc[:, 'tup'].to_dict()

    neighbours_dict = {
        k: tuple(x for x in v if zone_min < x < zone_max)
        for k, v in neighbours_dict.items() if zone_min < k < zone_max
        }
    return neighbours_dict


def _ringzone_dict(region):

    fp = pkg_resources.resource_filename(
        'tablesalt', 'resources/networktopodk/national_ringzone.csv'
        )

    ringzone = pd.read_csv(
        fp,
        index_col=0
        )
    zone_min, zone_max = REGION_ZONES[region]

    ringzone = ringzone.loc[
        str(zone_min + 1): str(zone_max - 1),
        str(zone_min + 1): str(zone_max - 1)
        ]
    ringzone = ringzone.stack()
    ringzone = ringzone.to_dict()
    ringzone = {(k[0], int(k[1])): int(v) for k, v in ringzone.items()}
    return ringzone

def _neighbour_edges(neighbours):

    n_edges = set()
    for k, v in neighbours.items():
        start_set = {(k, x) for x in v}
        n_edges.update(start_set)

    return n_edges


def _shared_neighbours(edge, ndict):

    node1 = set(ndict[edge[0]])
    node2 = set(ndict[edge[1]])
    return node1.intersection(node2)

def _load_tripzones(region_stops, region_zonemap):
    # if this doesn't exist download from rejseplan

    fp = pkg_resources.resource_filename(
    'tablesalt', 'resources/networktopodk/stop_times.txt'
    )
    stoptimes = pd.read_csv(
        fp, encoding='iso-8859-1',
        usecols=['trip_id', 'stop_id', 'stop_sequence']
        )

    stoptimes = stoptimes.query("stop_id in @region_stops").values
    counts = Counter(stoptimes[:, 0])
    twomin = [x for x, y in counts.items() if y >= 2]
    stoptimes = stoptimes[np.isin(stoptimes[:, 0], twomin)]
    stoptimes = stoptimes[np.lexsort((stoptimes[:, 2], stoptimes[:, 0]))]
    stopzones = [region_zonemap[x] for x in stoptimes[:, 1]]
    trip_zones = zip(stoptimes[:, 0], stopzones)

    return trip_zones

class ZoneGraph():
    """
    class for analysing the tariff zones
    as an undirected graph

    parameter
    ---------
    adjacency_array:
        an adjacency array for the the graph

    """
    PATH_CACHE = {}


    def __init__(
            self, 
            region: Optional[str] = 'sjælland', 
            mode: Optional[str] = None
            ) -> None:

        """


        Parameters
        ----------
        region : TYPE, optional
            DESCRIPTION. The default is None.
        mode : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """


        self.region = region
        self.data = EdgeMaker().make_edges(mode)

        self.columns = self.data['idx']
        self.rows = self.data['idx']
        self.rev_columns = self.data['rev_idx']
        self.rev_rows =  self.data['rev_idx']
        self.graph = nx.from_numpy_matrix(self.data['adj_array'])
        self._ringzone_dict = _ringzone_dict(self.region)
        self.SHORTEST_PATHS_CACHE = {}

    @classmethod
    def ring_dict(cls, region):

        return _ringzone_dict(region)

    def _as_graph(self):
        """return a networkx graph object"""
        return

    def find_paths(self, start, end, limit=None):
        """
        return a generator of simple paths
        between the start zone and the end zone
        limit the lenght of the path with limit
        """
        ring_dist = self._ringzone_dict[(start, end)]

        source = self.columns[start]
        target = self.rows[end]
        paths = nx.all_simple_paths(
            self.graph, source=source,
            target=target, cutoff=ring_dist
            )

        for found in paths:
            mapped = tuple(self.rev_columns[x] for x in found)
            self.PATH_CACHE[(start, end)] = mapped
            yield mapped

    def shortest_path(self, start, end):
        """
        return the shortest path between the
        chosen start and end zones using
        djikstra's method
        """

        if (start, end) in self.SHORTEST_PATHS_CACHE:
            return self.SHORTEST_PATHS_CACHE[(start, end)]

        source = self.columns[start]
        target = self.rows[end]
        sp = nx.all_shortest_paths(self.graph, source, target)
        mapped = tuple(tuple(self.rev_columns[x] for x in l) for l in list(sp))

        self.SHORTEST_PATHS_CACHE[(start, end)] = mapped

        return mapped

