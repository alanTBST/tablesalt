# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:08:07 2020

@author: alkj
"""


import json
import os
import pkg_resources
import zipfile
from io import BytesIO
from itertools import groupby, chain
from operator import itemgetter
from pathlib import Path
from typing import (
    AnyStr,
    Dict,
    Mapping,
    Optional,
    Union,
    Tuple,
    Set,
    Any,
    List
)

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import MultiPolygon, Point
from shapely import wkt

#from tablesalt.common import check_dw_for_table, insert_query, make_connection
from tablesalt.common.io import mappers

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
        if self._agency_ids is None:
            self.load_agency()
        return self._agency_ids

    @property
    def id_agency(self) -> Dict[int, str]:
        if self._id_agency is None:
            self.load_agency()
        return self._id_agency

    @property
    def route_agency(self) -> Dict[str, int]:
        if self._route_agency is None:
            self.load_routes()
        return self._route_agency

    @property
    def route_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self._route_shapes is None:
            trips = self.load_trips()
            self._process_trips(trips)
        return self._route_shapes

    @property
    def shape_lines(self) -> Dict[int, shapely.geometry.linestring.LineString]:
        if self._shape_linestrings is None:
            self._process_shapes(
                self.load_shapes(return_value=True)
                )
        return self._shape_linestrings

    @property
    def route_linestrings(
        self
        ) -> Dict[int, Tuple[shapely.geometry.linestring.LineString, ...]]:

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
        """
        Laod and return agency.txt data as a tuple of two
        dictionaries

        Returns
        -------
        id_agency : dict
            agency dict with the agency id as the key and the
            agency name as the value.
        agency_id : dict
            the agency name is the key and the values are tuples
            of the ids. Some agencies may have more than one id
        """
        if filepath is None:
            filepath = self.DEFAULT_GTFS_LOC
        agency = pd.read_csv(os.path.join(filepath, 'agency.txt'))
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
        filepath: FILE_PATH = None,
        return_value: Optional[bool] = False
        ) -> pd.core.frame.DataFrame:
        """
        Load the shapes.txt file as a dataframe

        Returns
        -------
        pandas.DataFrame
            A dataframe of the shape.txt data.
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
        """
        Load the routes.txt file as a dataframe

        Returns
        -------
        routes : pd.DataFrame
            A dataframe of the routes.txt data.
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
                'RejseplanenStoppesteder.csv'
                )
            )
    def __init__(self) -> None:

        pass


    def load_tariffzones(
        self,
        takst: Optional[str] = 'sjælland'
        ) -> gpd.geodataframe.GeoDataFrame:
        """
        Load the tariffzones geospatial data.

        Parameters
        ----------
        takst : str, optional


        Returns
        -------
        gdf : geopandas.GeoDataFrame
            a geodataframe of the tariffzones.
        """
        if not takst == 'sjælland':
            raise NotImplementedError(
                f"takst {takst} is not supported yet"
                )

        gdf = gpd.read_file(self.DEFAULT_ZONE_LOC)
        gdf.columns = [x.lower() for x in gdf.columns]
        gdf.loc[:, 'businesske'] = gdf.loc[:, 'businesske'].apply(
            lambda x: int('1'+ x.zfill(4)[1:]))

        gdf.rename(columns={'businesske': 'natzonenum'}, inplace=True)

        gdf = gdf.to_crs(epsg=4326)

        return gdf

    def load_stops_data(self) -> gpd.geodataframe.GeoDataFrame:
        """
        Load and return a dataframe of stops in Denmark.

        Returns
        -------
        stops_gdf : geopandas.GeoDataFrame
            a geo dataframe of rejseplan stop places.

        """
        fp = self.DEFAULT_STOPS_LOC
        try:
            with open(fp, 'r') as f:
                date = f.readline()
        except UnicodeDecodeError:
            with open(fp, 'r', encoding='iso-8859-1') as f:
                date = f.readline()
        if 'period' in date.lower() or 'export' in date.lower():
            skiprows = 1
        else:
            skiprows = 0
        stops_df = pd.read_csv(
            fp, header=None,
            skiprows=skiprows,
            encoding='iso-8859-1',
            sep=';'
            )
        stops_df.columns = ['UIC', 'Name', 'long_utm32N', 'lat_utm32N']

        stops_gdf = gpd.GeoDataFrame(
            stops_df,
            geometry=gpd.points_from_xy(
                stops_df.iloc[:, 2], stops_df.iloc[:, 3]
                )
            )

        stops_gdf.crs = "EPSG:32632" # this is the utm32N given by rejsedata
        #if not projection:
         #   projection = "EPSG:32632"
        stops_gdf = stops_gdf.to_crs(epsg=4326)

        return stops_gdf

    @staticmethod
    def _set_stog_location(df):

        s_stops = mappers['s_uic']
        corr_s_stops = [x - 90000 for x in s_stops]
        corr_stops = df.query("UIC in @corr_s_stops").copy(deep=True)
        corr_stops.loc[:, 'UIC'] = corr_stops.loc[:, 'UIC'] + 90_000
        out_frame = pd.concat([df, corr_stops])

        return out_frame

    def stop_zone_map(
        self,
        region: Optional[str] = 'sjælland'
        ) -> Dict[int, int]:

        stops = self.load_stops_data()
        stops = self._set_stog_location(stops)
        zones = self.load_tariffzones(takst=region)
        joined = gpd.sjoin(stops, zones)

        return dict(
            zip(joined.loc[:, 'UIC'],  joined.loc[:, 'natzonenum'])
            )
    @staticmethod
    def _neighbour_dict(region):
        """Load and convert neighbours dset to dict (adj list)"""

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
            for k, v in neighbours_dict.items() if zone_min < k < zone_max
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


class EdgeMaker(_GTFSloader):

    def __init__(self) -> None:
        super().__init__()
        self.zones = TakstZones()


    def _shape_proc(self, mode: Optional[str] = None):

        zones = self.zones.load_tariffzones()
        shape_frame = self.shapes_to_gdf()
        if mode is not None:
            wanted_routes = self.route_types(str)
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
    def _edges_to_array(edges):

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
        start_point: shapely.geometry.point.Point,
        zone_polygons: Dict[int, shapely.geometry.polygon.Polygon]
        ) -> int:

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
        zone_polygons: Dict[int, shapely.geometry.polygon.Polygon],
        points: List[shapely.geometry.point.Point],
        neigh_dict: Dict[int, Tuple[int]]
        ) -> Set[Tuple[int, ...]]:

        finished_zones = set()
        edges = set()
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


    def make_edges(self, mode: Optional[str] = None):

        neigh_dict = self.zones._neighbour_dict('sjælland') # for now
        shape_zones, zone_polys = self._shape_proc(mode)

        all_shape_edges = {}
        for shape_id in shape_zones:

            zone_ids = set(shape_zones[shape_id]) # v
            zone_id_poly = {k: v for k, v in zone_polys.items() if k in zone_ids}
            lstring = self.shape_lines[shape_id]
            as_points = [Point(x) for x in lstring.coords]

            init_point = as_points[0]
            try:
                start_zone = self._get_start_zone(init_point, zone_id_poly)
            except ValueError:
                continue
            shape_edges = self._find_shape_edges(
                start_zone, zone_ids, zone_id_poly, as_points, neigh_dict
                )
            all_shape_edges[shape_id] = shape_edges

        all_edges = set.union(*all_shape_edges.values())

        return self._edges_to_array(all_edges)


# def create_stops_spatial_table(stops_df, schema='Rejseplanen'):
#     """
#     Make a spatial table in MS SQL Server for rplan stops.

#     Parameters
#     ----------
#     stops_df : pandas.DataFrame
#         DESCRIPTION.

#     Returns
#     -------
#     badqs : list
#         a list of the records that could not be created
#         in the data warehouse

#     """
#     createquery = (
#         f"CREATE TABLE [{schema}].[StoppeSteder] ("
#         "uic int, station nvarchar(50), lat_utm32N int, "
#         "long_utm32N int, "
#         "geom geography"
#         ")"
#         )

#     tups = stops_df.loc[:, (
#         'UIC', 'Name', 'lat_utm32N',
#         'long_utm32N', 'geometry'
#         )]

#     tups.loc[:, 'geometry'] = \
#         tups.loc[:, 'geometry'].apply(
#             lambda x: f"geography::STGeomFromText('{x.to_wkt()}', 32632)"
#             )
#     tups = tups.itertuples(name=None, index=False)
#     qry = insert_query('Rejseplanen', 'stops', 5)

#     badqs = []
#     with make_connection() as con:
#         cursor = con.cursor()
#         try:
#             cursor.execute(createquery)
#             con.commit()
#         except:
#             pass
#         for tup in tups:
#             try:
#                 cursor.execute(qry, tup)
#             except Exception:
#                 badqs.append(tup)
#         con.commit()
#         cursor.close()
#     return badqs


# def create_zonemap_table(mapdf, schema='Rejseplanen'):
#     """
#     Make zonemap table in MS SQL Server.

#     Parameters
#     ----------
#     mapdf : pandas.DataFrame
#         DESCRIPTION.
#     schema : str, optional
#         the db schema to use.
#         The default is 'Rejseplanen'.

#     Returns
#     -------
#     None.

#     """
#     createquery = (
#         f"CREATE TABLE [{schema}].[stopzonemap] ("
#         "uic int, natzonenum int)"
#         )
#     tups = mapdf.itertuples(index=False, name=None)

#     with make_connection() as con:
#         cursor = con.cursor()
#         cursor.execute(createquery)
#         con.commit()
#         # cursor.commit()
#         cursor.executemany(
#             f"INSERT INTO [{schema}].[stopzonemap] VALUES (?, ?)",
#             tups
#             )
#         con.commit()
#         cursor.close()


# def create_zones_spatial_table(tzones_df, schema='Rejseplanen'):
#     """
#     Create the tariff zones spatial table in MS SQL Server.

#     Parameters
#     ----------
#     tzones_df : geopandas.GeoDataFrame
#         geodataframe of the national tariff zones.
#     schema : str, optional
#         the schema to save the table in.
#         The default is 'Rejseplanen'.

#     Returns
#     -------
#     None.

#     """
#     createquery = (f"CREATE TABLE [{schema}].[zones] ("
#                    "operator_id int, natzonenum int, "
#                    "geom geography)")
#     with make_connection() as con:
#         cursor = con.cursor()
#         cursor.execute(createquery)
#         con.commit()

#     tups = tzones_df.loc[:, (
#         'operator', 'natzonenum',
#         'geom_transformed'
#         )]

#     tups.loc[:, 'ismulti'] = \
#         tups.loc[:, 'geom_transformed'].apply(
#             lambda x: isinstance(x, MultiPolygon))
#     multipoly = tups.loc[tups.loc[:, 'ismulti'] is True]
#     poly = tups.loc[tups.loc[:, 'ismulti'] is False]
#     multipoly = multipoly.drop('ismulti', axis=1)
#     poly = poly.drop('ismulti', axis=1)

#     poly.loc[:, 'geom_transformed'] = \
#         poly.loc[:, 'geom_transformed'].apply(
#             lambda x: f"geography::STPolyFromText('{x.to_wkt()}', 32632)")

#     multipoly.loc[:, 'geom_transformed'] = \
#         multipoly.loc[:, 'geom_transformed'].apply(
#             lambda x: f"geography::STMPolyFromText('{x.to_wkt()}', 32632)")

#     poly_tups = poly.itertuples(name=None, index=False)
#     multipoly_tups = multipoly.itertuples(name=None, index=False)

#     # qry = _insert_query(schema, 'tariff_zones', 3)
#     qry = lambda x: (
#         f"INSERT INTO [{schema}].[tariff_zones] "
#         f"VALUES ({x[0]}, '{x[1]}', {x[2]})"
#         )

#     badqs = []
#     with make_connection() as con:
#         cursor = con.cursor()
#         for tup in poly_tups:
#             try:
#                 cursor.execute(qry(tup))
#             except Exception:
#                 badqs.append((tup, qry(tup)))
#         con.commit()
#         cursor.close()

#     with make_connection() as con:
#         cursor = con.cursor()
#         for tup in multipoly_tups:
#             try:
#                 cursor.execute(qry(tup))
#             except Exception:
#                 badqs.append((tup, qry(tup)))
#         con.commit()
#         cursor.close()

#     return badqs
