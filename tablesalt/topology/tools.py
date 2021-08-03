# -*- coding: utf-8 -*-
"""
Classes to download and manipulate routing and spatial data
"""


import json
import os
from tablesalt.topology.stopnetwork import StopsList
import zipfile
from functools import singledispatch
from io import BytesIO
from itertools import chain, groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, AnyStr, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pkg_resources
import requests  # type: ignore
import shapely  # type: ignore
from shapely import wkt  # type: ignore
from shapely.geometry.linestring import LineString  # type: ignore
from shapely.geometry.point import Point # type: ignore
from shapely.geometry.polygon import Polygon  # type: ignore
from tablesalt.common.io import mappers
from tablesalt.topology.stopnetwork import StopsList


FILE_PATH = Union[str, bytes, 'os.PathLike[Any]']

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

@singledispatch
def determine_takst_region(zone_sequence):

    return zone_sequence


@determine_takst_region.register
def _(zone_sequence: int) -> str:

    if zone_sequence < 1100:
        return "movia_h"
    if 1100 < zone_sequence <= 1200:
        return "movia_v"
    if 1200 < zone_sequence < 1300:
        return "movia_s"

@determine_takst_region.register
def _(zone_sequence: tuple) -> str:

    if all(x < 1100 for x in zone_sequence):
        return "movia_h"
    if all(1100 < x <= 1200 for x in zone_sequence):
        return "movia_v"
    if all(1200 < x < 1300 for x in zone_sequence):
        return "movia_s"
    return "dsb"

@determine_takst_region.register
def _(zone_sequence: list) -> str:

    if all(x < 1100 for x in zone_sequence):
        return "movia_h"
    if all(1100 < x <= 1200 for x in zone_sequence):
        return "movia_v"
    if all(1200 < x < 1300 for x in zone_sequence):
        return "movia_s"
    return "dsb"
@determine_takst_region.register
def _(zone_sequence: set) -> str:

    if all(x < 1100 for x in zone_sequence):
        return "movia_h"
    if all(1100 < x <= 1200 for x in zone_sequence):
        return "movia_v"
    if all(1200 < x < 1300 for x in zone_sequence):
        return "movia_s"
    return "dsb"
    
class _StopLoader:

    DEFAULT_STOPS_LOC = pkg_resources.resource_filename(
        'tablesalt',
        os.path.join(
            'resources',
            'networktopodk',
            'RejseplanenStoppesteder.csv'
            )
        )

    DEFAULT_ZONE_LOC = pkg_resources.resource_filename(
            'tablesalt',
            os.path.join(
                'resources', 'networktopodk',
                'DKTariffZones', 'takstsjaelland',
                'sjaelland_zones.shp'
                )
            )

    def __init__(self):
        pass

    def _load_zones(self):

        return

    def _load_stops(self):

        return


class _GTFSloader:

    DEFAULT_GTFS_LOC = pkg_resources.resource_filename(
        'tablesalt',
        os.path.join(
            'resources', 'networktopodk', 'gtfs'
            )
        )
    GTFS_ROUTE_TYPES = {
        0: 'light_rail',
        109: 'suburban_rail',
        1: 'metro_rail',
        400: 'metro_rail',
        2: 'rail',
        100: 'rail',
        3: 'bus',
        700: 'bus',
        715: 'demand_bus',
        4: 'ferry'
        }

    def __init__(self) -> None:
        """
        Class to load gtfs data
        """

        self._route_agency = None
        self._route_shapes = None
        self._id_agency = None
        self._agency_ids = None
        self._route_types = None
        self._route_shapes = None
        self._shape_linestrings = None
        self._route_linestrings = None

    @property
    def agency_ids(self) -> Dict[str, Tuple[int, ...]]:
        """return the agency -> id mapping

        :return: a dictionary with the agency name as key and it's id as value
        :rtype: Dict[str, Tuple[int, ...]]
        """
        if self._agency_ids is None:
            self.load_agency()
        return self._agency_ids

    @property
    def id_agency(self) -> Dict[int, str]:
        """return the id -> agency mapping

        :return: a dictionary with the agency id as key and its name as value
        :rtype: Dict[int, str]
        """
        if self._id_agency is None:
            self.load_agency()
        return self._id_agency

    @property
    def route_agency(self) -> Dict[str, int]:
        """return the route id -> agency id mapping

        :return: a dictionary with the route id as key and the agency id as values
        :rtype: Dict[str, int]
        """
        if self._route_agency is None:
            self.load_routes()
        return self._route_agency

    @property
    def route_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """return the route id -> shapeid mapping

        :return: a dictionary with the route id as keys and a tuple of the
        corresponding shape ids as values
        :rtype: Dict[str, Tuple[int, ...]]
        """
        if self._route_shapes is None:
            trips = self.load_trips()
            self._process_trips(trips)
        return self._route_shapes

    @property
    def shape_lines(self) -> Dict[int, LineString]:
        """return the shape id -> linestring mapping

        :return: a dictionary with the shape id as key and linestrings values
        :rtype: Dict[int, LineString]
        """
        if self._shape_linestrings is None:
            self._process_shapes(
                self.load_shapes(return_value=True)
                )
        return self._shape_linestrings

    @property
    def route_linestrings(
        self
        ) -> Dict[int, Tuple[LineString, ...]]:
        """return the route names and linestrings dictionary

        :return: a dictionary with the route names as keys and a tuple
            of shapely Linestrings as values
        :rtype: Dict[int, Tuple[LineString, ...]]
        """
        if self._route_linestrings is None:
            shape_lines = self.shape_lines
            self._route_linestrings = {
                k: tuple(shape_lines[x] for x in v)
                for k, v in self.route_shapes.items()
                }
        return self._route_linestrings

    def route_types(self, vals):
        if self._route_types is None:
            self.load_routes()
        if vals == int:
            return self._route_types
        if vals == str:
            return {k: self.GTFS_ROUTE_TYPES.get(v, 'unknown') for
                    k, v in self._route_types.items()}
        raise TypeError("only int and str supported")

    # change this from: no return_value = true
    def load_agency(
        self,
        filepath: Optional[AnyStr] = None,
        return_value: Optional[bool] = False
        ) -> Tuple[Dict[int, str], Dict[str, Tuple[int, ...]]]:
        """Load and return agency.txt data as a tuple of two
        dictionaries




        :param filepath: the path to the agency.txt file if not using the default, defaults to None
        :type filepath: Optional[AnyStr], optional
        :param return_value: whether to return a value or not, defaults to False
        :type return_value: Optional[bool], optional
        :return:{'id_agency': {agency_id: agency_name}
                {'agency_id': {agency_name: tuple(agency_id}}
        :rtype: Tuple[Dict[int, str], Dict[str, Tuple[int, ...]]]
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC

        fp = Path(filepath) / 'agency.txt'
        agency = pd.read_csv(fp)
        id_agency = dict(
            zip(agency.loc[:, 'agency_id'], agency.loc[:, 'agency_name'])
            )
        self._id_agency = id_agency

        agency = zip(agency.loc[:, 'agency_name'],
                    agency.loc[:, 'agency_id'])
        agency = sorted(agency, key=lambda x: x[0])

        agency_ids = {
            k: tuple(x[1] for x in g) for
            k, g in groupby(agency, key=lambda y: y[0])
            }
        self._agency_ids = agency_ids

        if return_value:
            return {
                'agency_ids': agency_ids,
                'id_agency': id_agency
            }

    def load_shapes(
        self,
        filepath: Optional[FILE_PATH] = None,
        return_value: Optional[bool] = False
        ) -> pd.core.frame.DataFrame:
        """load the shapes.txt data into a dataframe

        :param filepath: the path to the  defaults to None
        :type filepath: Optional[FILE_PATH], optional
        :param return_value: [description], defaults to False
        :type return_value: Optional[bool], optional
        :return: [description]
        :rtype: pd.core.frame.DataFrame
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC
        shapes =  pd.read_csv(
            os.path.join(filepath, 'shapes.txt'),
            low_memory=False
            )
        if return_value:
            return shapes

    def _process_shapes(
        self,
        shapes: pd.core.frame.DataFrame
        ) -> None:
        """create the self._shape_linestrings attribute

        :param shapes: [description]
        :type shapes: pd.core.frame.DataFrame
        """

        shapes = shapes.sort_values(['shape_id', 'shape_pt_sequence'])
        shapes = shapes.itertuples(name=None, index=False)

        shape_points = {
            key: str(
                tuple(
                    str(x[2]) + ' ' + str(x[1]) for x in grp
                    )
                ) for
            key, grp in groupby(shapes, key=itemgetter(0))
            }

        shape_points = {
            k: v.replace("'", "") for
            k, v in shape_points.items()
            }

        shape_points = {
            k: 'LINESTRING ' + v for
            k, v in shape_points.items()
            }

        shape_lines = {
            k: wkt.loads(v) for k, v in shape_points.items()
            }

        self._shape_linestrings = shape_lines

    def shapes_to_gdf(self) -> gpd.geodataframe.GeoDataFrame:
        """convert the shapes point data to a geodataframe

        :return: a geopandas geodataframe of the shapes data
        :rtype: gpd.geodataframe.GeoDataFrame
        """

        route_shapes = self.route_shapes
        rs_frame = pd.DataFrame.from_dict(
            route_shapes, orient='index'
            ).stack()
        rs_frame = rs_frame.reset_index()
        rs_frame.columns = ['route_id', 'shp', 'shape_id']

        rs_frame.loc[:, 'geometry'] = \
            rs_frame.loc[:, 'shape_id'].map(self.shape_lines)

        gdf = gpd.GeoDataFrame(
            rs_frame, geometry='geometry', crs="epsg:4326"
            )

        return gdf

    def load_routes(
        self,
        filepath: Optional[AnyStr] = None,
        return_value: Optional[bool] = False
        ) -> pd.core.frame.DataFrame:
        """Load the routes.txt file as a dataframe

        :param filepath: the path to the routes.txt if not using the default,
            defaults to None
        :type filepath: Optional[AnyStr], optional
        :param return_value: to return data or not, defaults to False
        :type return_value: Optional[bool], optional
        :return: a dataframe of the routes data
        :rtype: pd.core.frame.DataFrame
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC

        routes = pd.read_csv(
            os.path.join(filepath, 'routes.txt'),
            low_memory=False,
            usecols=[
                'route_id',
                'agency_id',
                'route_short_name',
                'route_long_name',
                'route_type'
                ]
            )
        self._route_agency = dict(
            zip(routes.loc[:, 'route_id'], routes.loc[:, 'agency_id'])
        )

        self._route_types = dict(
            zip(routes.loc[:, 'route_id'], routes.loc[:, 'route_type'])
        )
        if return_value:

            return routes

    def load_trips(
        self,
        filepath: Optional[AnyStr] = None
        ) -> pd.core.frame.DataFrame:
        """load the trips.txt data

        :param filepath: path to trips.txt if not using default, defaults to None
        :type filepath: Optional[AnyStr], optional
        :return: a trips dataframe
        :rtype: pd.core.frame.DataFrame
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC

        trips = pd.read_csv(
            os.path.join(filepath, 'trips.txt'),
            low_memory=False,
#            usecols=['route_id', 'shape_id']
            )
        self._trip_route = dict(
            zip(trips.loc[:, 'trip_id'], trips.loc[:, 'route_id'])
            )

        return trips

    def _process_trips(
            self,
            trips: pd.core.frame.DataFrame
            ) -> None:
        """process the trips dataframe

        :param trips: a dataframe loaded with self.load_trips
        :type trips: pd.core.frame.DataFrame
        """
        trips = trips.fillna(0)
        trips = trips.drop_duplicates()
        trips = trips[['route_id', 'shape_id']]
        trips = trips.sort_values(['route_id', 'shape_id'])
        trips = trips.itertuples(name=None, index=False)

        route_to_shapes = {
            key: tuple(set(int(x[1]) for x in grp if x[1] > 0)) for
            key, grp in groupby(trips, key=itemgetter(0))
            }
        self._route_shapes = route_to_shapes


    def _load_stop_times(
        self,
        filepath: Optional[AnyStr] = None
        ) -> pd.core.frame.DataFrame:
        """load the stoptimes.txt data

        :param filepath: [description], defaults to None
        :type filepath: Optional[AnyStr], optional
        :return: [description]
        :rtype: pd.core.frame.DataFrame
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC

        stoptimes = pd.read_csv(
            os.path.join(filepath, 'stop_times.txt'),
            dtype={'trip_id': int, 'stop_id': int, 'stop_sequence': int},
            usecols=['trip_id', 'stop_id', 'stop_sequence']
        )
        stoptimes = stoptimes.sort_values(['trip_id', 'stop_sequence'])

        stop_sequence = {
            key: tuple(x[1] for x in grp) for key, grp in
            groupby(stoptimes.itertuples(name=None, index=False), key=itemgetter(0))
        }


        return stoptimes


class TakstZones:

    DEFAULT_ZONE_LOC = pkg_resources.resource_filename(
            'tablesalt',
            os.path.join(
                'resources', 'networktopodk',
                'DKTariffZones', 'takstsjaelland',
                'sjaelland_zones.shp'
                )
            )

    DEFAULT_STOPS_LOC = pkg_resources.resource_filename(
            'tablesalt',
            os.path.join(
                'resources',
                'networktopodk',
                'stops.json'
                )
            )
    def __init__(self) -> None:
        """
        A class for interacting with the TakstZone spatial data

        :return: ''
        :rtype: None

        """
        self._list_of_stops = StopsList(self.DEFAULT_STOPS_LOC)


    def stop_zone_map(
        self,
        ) -> Dict[int, int]:
        """
        Return a mapping of stopids to zone ids

        :param region: the region to get a map for, defaults to 'sjælland'
        :type region: Optional[str], optional
        :return: a dictionary with stopids as keys and zoneids as values
        :rtype: Dict[int, int]

        """

        stops = self.stop_geodataframe()
        stops = self._set_stog_location(stops)
        zones = self.load_tariffzones()
        joined = gpd.sjoin(stops, zones)
        # NOTE: decide on whether border stations should be added here
        return dict(
            zip(joined.loc[:, 'stop_id'],  joined.loc[:, 'natzonenum'])
            )

    def load_tariffzones(
        self,
        ) -> gpd.geodataframe.GeoDataFrame:
        """
        Load the tariffzones geospatial data.

        :param takst: the takstset to load, defaults to 'sjælland'
        :type takst: Optional[str], optional
        :raises NotImplementedError: if the region is not supported
        :return: a geodataframe of the tariffzones
        :rtype: geopandas.GeoDataFrame

        """

        gdf = gpd.read_file(self.DEFAULT_ZONE_LOC)
        gdf.columns = [x.lower() for x in gdf.columns]
        gdf.loc[:, 'businesske'] = gdf.loc[:, 'businesske'].apply(
            lambda x: int('1'+ x.zfill(4)[1:])
            )

        gdf.rename(columns={'businesske': 'natzonenum'}, inplace=True)

        gdf = gdf.to_crs(epsg=4326) # set projection to wgs84

        return gdf

    def stop_geodataframe(self) -> gpd.geodataframe.GeoDataFrame:
        """
        Load and return a dataframe of stops in Denmark.

        :return: a geo dataframe of rejseplan stop places
        :rtype: geopandas.GeoDataFrame

        """
        stops_df = pd.DataFrame(self._list_of_stops)

        stops_gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=gpd.points_from_xy(
                stops_df.iloc[:, 5], stops_df.iloc[:, 4]
                )
            )
        stops_gdf.crs = 4326  #set projection WGS84 

        return stops_gdf

    @staticmethod
    def _set_stog_location(stops_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        """create the data for stog stations ie UIC = 869xxxx

        :param stops_df: dataframe
        :type stops_df: pd.core.frame.DataFrame
        :return: stops data
        :rtype: pd.core.frame.DataFrame
        """

        s_stops = mappers['s_uic']
        corr_s_stops = [x - 90000 for x in s_stops]
        corr_stops = stops_df.query("stop_id in @corr_s_stops").copy(deep=True)
        corr_stops.loc[:, 'stop_id'] = corr_stops.loc[:, 'stop_id'] + 90_000
        out_frame = pd.concat([stops_df, corr_stops])

        return out_frame

    @staticmethod
    def neighbour_dict(region: str) -> Dict[int, Any]:
        """Load and convert neighbours dset to a dictionary"""

        fp = pkg_resources.resource_filename(
            'tablesalt', 'resources/networktopodk/national_neighbours.csv'
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
            for k, v in neighbours_dict.items() if
            zone_min < k < zone_max
            }
        return neighbours_dict


class RejseplanLoader(_GTFSloader):

    def __init__(self, **urls: str) -> None:
        super().__init__()

        self.urls = dict(**urls)
        if not self.urls:
            self.urls = self._load_default_rejseplan_url()
        self._assert_zip()

        self._responses = self._get_responses()
        self._content = {}

    @staticmethod
    def _load_default_rejseplan_url() -> Dict[str, str]:

        conf_fp = pkg_resources.resource_filename(
            'tablesalt',
            os.path.join(
                'resources', 'config',
                'rejseplan_url.json'
                )
            )
        with open(conf_fp, 'r') as fp:
            asdict = json.load(fp)
        return asdict

    def _assert_zip(self) -> None:
        for _, v in self.urls.items():
            if 'zip' not in v:
                raise ValueError("urls must be .zip files")

    def _get_responses(self):
        responses = {}
        for k, v in self.urls.items():
            responses[k] = requests.get(v)
        return responses

    def _get_zip_content(self) -> Dict[str, Tuple[str, ...]]:

        for k, v in self._responses.items():
            self._content[k] = zipfile.ZipFile(
                BytesIO(v.content)
                ).namelist()
                
    @property
    def contents(self):
        if not self._content:
            self._get_zip_content()
        return self._content
    """
    @staticmethod
    def download_new_stops_data(url: Optional[AnyStr]) -> None:
        Retrieve and extract the rejseplan stoppestedder data.

        _path = pkg_resources.resource_filename(
            'tablesalt',
            os.path.join(
                'resources', 'networktopodk'
                )
            )

        stops_url = _load_rejseplan_url()['stops_url']
        response = requests.get(stops_url, stream=True)
        with zipfile.ZipFile(BytesIO(response.content)) as my_zip_file:
            for x in my_zip_file.namelist():
                my_zip_file.extract(x, path=_path)
    """
    def download_data(self) -> None:

        return


class EdgeMaker():

    def __init__(self) -> None:
        """
        Class to create edges for a network graph from GTFS data

        :return: ''
        :rtype: None

        """

        super().__init__()
        self.zones = TakstZones()
        self.loader = _GTFSloader()

    def _shape_proc(
            self,
            mode: Optional[str] = None
            ) -> Tuple[Dict[int, Tuple[int, ...]], Dict[int, Polygon]]:
        """[summary]

        :param mode: [description], defaults to None
        :type mode: Optional[str], optional
        :return: [description]
        :rtype: Tuple[Dict[int, Tuple[int, ...]], Dict[int, Polygon]]
        """

        zones = self.zones.load_tariffzones()
        shape_frame = self.loader.shapes_to_gdf()

        if mode is not None:
            wanted_routes = self.loader.route_types(str)
            wanted_routes = {k for k, v in wanted_routes.items() if mode in v}
            shape_frame = shape_frame.query("route_id in @wanted_routes")

        joined = gpd.sjoin(zones, shape_frame)

        shape_zones = zip(joined.loc[:, 'shape_id'].astype(int),
                          joined.loc[:, 'natzonenum'])
        shape_zones = sorted(shape_zones, key=itemgetter(0))

        shape_zones = {
            key: tuple(x[1] for x in grp) for key, grp in
            groupby(shape_zones, key=itemgetter(0))
            }

        shape_zones = {
            k: v for k, v in shape_zones.items() if len(v) > 1
            }

        zone_polys = dict(
            zip(joined.loc[:, 'natzonenum'],
                joined.loc[:, 'geometry'])
            )

        return shape_zones, zone_polys
    @staticmethod
    def _edges_to_array(edges: Set[Tuple[int, int]]) ->  Dict[str, Union[np.ndarray, Dict[int, int]]]:
        """return the output for the ZoneGraph class in zonegraph

        :param edges: edges found for all shapes
        :type edges: Set[Tuple[int, int]]
        :return: a dictionary of the adjacency array for a graph and mappings
            for the array columns to zone numbers
        :rtype: Dict[str, Union[np.ndarray, Dict[int, int]]]
        """

        array_indices = sorted(set(chain(*edges)))
        idx = {j:i for i, j in enumerate(array_indices)}
        rev_idx = {v:k for k, v in idx.items()}

        array = np.zeros(
            (len(array_indices), len(array_indices)), int
            )

        for edge in edges:
            y = idx[edge[0]]
            x = idx[edge[1]]
            array[x, y] = 1

        return {'adj_array': array,
                'idx': idx,
                'rev_idx': rev_idx}
    @staticmethod
    def _get_start_zone(
        start_point: Point,
        zone_polygons: Dict[int, Polygon]
        ) -> int:
        """find the starting zone of the first point of a Linestring

        :param start_point: the starting point to check
        :type start_point: Point
        :param zone_polygons: dictionary of zone polygons
        :type zone_polygons: Dict[int, Polygon]
        :raises ValueError: if the starting point is not in any of the given zones
        :return: the zone number of the starting point
        :rtype: int
        """
        for zone, poly in zone_polygons.items():
            pinp = start_point.within(poly)
            if pinp:
                start_zone = zone
                return start_zone

        raise ValueError(f"start_point {start_point} not in given zones")

    @staticmethod
    def _find_shape_edges(
        start_zone: int,
        zone_ids: Set[int],
        zone_polygons: Dict[int, Polygon],
        points: List[shapely.geometry.point.Point],
        neigh_dict: Dict[int, Tuple[int]]
        ) -> Set[Tuple[int, ...]]:
        """find the edges created by a list of points in a shape/line

        :param start_zone: the starting zone
        :type start_zone: int
        :param zone_ids: the set zone numbers in the shape
        :type zone_ids: Set[int]
        :param zone_polygons: a dictionary of shapely polygons for each zone number
        :type zone_polygons: Dict[int, Polygon]
        :param points: a list of shapely points
        :type points: List[shapely.geometry.point.Point]
        :param neigh_dict: hte dictionary of neighbour zones
        :type neigh_dict: Dict[int, Tuple[int]]
        :return: return the set of edges create by the list of points
        :rtype: Set[Tuple[int, ...]]
        """

        finished_zones = set()
        edges: Set[Tuple[int, ...]] = set()
        current_zone = start_zone
        for pt in points:
            if pt.within(zone_polygons[current_zone]):
                continue

            finished_zones.add(current_zone)
            if finished_zones == zone_ids:
                break

            neighbours = [
                x for x in neigh_dict[current_zone] if x in zone_ids
                ]
            for n in neighbours:
                if pt.within(zone_polygons[n]):
                    edge = (current_zone, n)
                    edges.add(edge)
                    current_zone = n
                    break

        return edges


    def make_edges(
            self,
            mode: Optional[str] = None
            ) -> Dict[str, Union[np.ndarray, Dict[int, int]]]:
        """create a set of edges for the zone graph

        :param mode: 'rail' or 'bus', defaults to None
        :type mode: Optional[str], optional
        :return: a dictionary with data for the zone graph
            {'adj_array': an adjacency array for the networkx graph,
            'idx': dictionary of mapping zone numbers to col idxs,
            'rev_idx': a reverse mapping of col idxs to zone numbers}
        :rtype: Dict[str, Union[np.ndarray, Dict[int, int]]]
        """
        neigh_dict = self.zones.neighbour_dict('sjælland') # for now
        shape_zones, zone_polys = self._shape_proc(mode)

        all_shape_edges = {}
        for shape_id in shape_zones:

            zone_ids = set(shape_zones[shape_id]) # v
            zone_id_poly = {k: v for k, v in zone_polys.items() if k in zone_ids}
            lstring = self.loader.shape_lines[shape_id]
            as_points = [Point(x) for x in lstring.coords]

            init_point = as_points[0]
            try:
                start_zone = self._get_start_zone(
                    init_point, zone_id_poly
                    )
            except ValueError:
                continue
            shape_edges = self._find_shape_edges(
                start_zone,
                zone_ids,
                zone_id_poly,
                as_points,
                neigh_dict
                )
            all_shape_edges[shape_id] = shape_edges

        all_edges = set.union(*all_shape_edges.values())

        return self._edges_to_array(all_edges)
