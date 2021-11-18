# -*- coding: utf-8 -*-
"""
Class to create an undirected graph
of the tariffzones in Denmark
"""

from typing import Dict, Generator, Optional, Tuple

import networkx as nx #type: ignore
import pandas as pd #type: ignore
import pkg_resources

from tablesalt.topology.tools import EdgeMaker

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

def _ringzone_dict(region: str) -> Dict[Tuple[int, int], int]:
    """load the ringzone csv file from the package and convert it
    to a dictionary

    :param region: the region or subregion to include
    :type region: str
    :return: a dictionary with a two tuple of zones
        and the distance in zones as values
    :rtype: Dict[Tuple[int, int], int]
    """
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


class ZoneGraph():

    PATH_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def __init__(
            self,
            region: Optional[str] = 'sjælland',
            mode: Optional[str] = None
            ) -> None:
        """
        A network graph that uses zones as nodes and
        a route connection as edges

        :param region: the region to create a graph for, defaults to 'sjælland'
        :type region: Optional[str], optional
        :param mode: the mode ['rail', 'bus', None], defaults to None
            None will produce a composite graph of rail and bus
        :type mode: Optional[str], optional
        :return: ''
        :rtype: None

        """

        self.region = region
        self.mode = mode
        self.data = EdgeMaker().make_edges(self.mode)
        self.columns = self.rows = self.data['idx']
        self.rev_columns = self.rev_rows = self.data['rev_idx']
        self.graph = nx.from_numpy_matrix(self.data['adj_array'])
        self._ringzone_dict = _ringzone_dict(self.region)
        self.SHORTEST_PATHS_CACHE: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    # OPTION put this in the TakstZones class in tools
    @classmethod
    def ring_dict(cls, region: str):
        """return the ringzone dictionary for the given region

        :param region: the region wanted:
             ['sjælland', 'hovedstaden', 'sydsjælland', 'vestsjælland']
        :type region: str
        :return: the ringzone dictionary
        :rtype: Dict[Tuple[int, int], int]
        """

        return _ringzone_dict(region)

    def _as_graph(self):
        """return a networkx graph object"""
        return self.graph

    def find_paths(
            self,
            start: int,
            end: int,
            ) -> Generator[Tuple[int, ...], None, None]:
        """
        Return all simple paths between the given start and end zones

        :param start: the start zone (national zone format)
        :type start: int
        :param end: the end zone (national zone format)
        :type end: int
        :yield: a tuple of the zone path
        :rtype: Generator[Tuple[int, ...], None, None]

        """

        ring_dist = self._ringzone_dict[(start, end)]

        source = self.columns[start]
        target = self.rows[end]
        paths = nx.all_simple_paths(
            self.graph,
            source=source,
            target=target,
            cutoff=ring_dist
            )

        for found in paths:
            mapped = tuple(self.rev_columns[x] for x in found)
            self.PATH_CACHE[(start, end)] = mapped
            yield mapped

    def shortest_path(
            self,
            start: int,
            end: int
            ) -> Tuple[Tuple[int, ...], ...]:
        """
        Return the shortest paths between the given start and
        end zones using Djikstra's method

        :param start: the starting zone (national zone format)
        :type start: int
        :param end: the ending zone (national zone format)
        :type end: int
        :return: all of the shortest zone paths
        :rtype: Tuple[Tuple[int, ...], ...]

        """

        if (start, end) in self.SHORTEST_PATHS_CACHE:
            return self.SHORTEST_PATHS_CACHE[(start, end)]

        source = self.columns[start]
        target = self.rows[end]
        sp = nx.all_shortest_paths(self.graph, source, target)
        mapped = tuple(
            tuple(self.rev_columns[x] for x in l) for l in list(sp)
            )

        self.SHORTEST_PATHS_CACHE[(start, end)] = mapped

        return mapped
